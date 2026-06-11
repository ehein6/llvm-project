//===- ViewOpGraph.cpp - View/write op graphviz graphs --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ViewOpGraph.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include <map>
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_VIEWOPGRAPHPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static const StringRef kLineStyleControlFlow = "dashed";
static const StringRef kLineStyleDataFlow = "solid";
static const StringRef kShapeNode = "Mrecord";
static const StringRef kShapeNone = "plain";

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (std::optional<int64_t> limit =
          OpPrintingFlags().getLargeElementsAttrLimit())
    return *limit;
  return 16;
}

/// Return all values printed onto a stream as a string.
static std::string strFromOs(function_ref<void(raw_ostream &)> func) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  func(os);
  return buf;
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str) {
  return "\"" + str + "\"";
}

/// For Graphviz record nodes:
/// " Braces, vertical bars and angle brackets must be escaped with a backslash
/// character if you wish them to appear as a literal character "
static std::string escapeLabelString(const std::string &str) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  for (char c : str) {
    if (llvm::is_contained({'{', '|', '<', '}', '>', '\n', '"'}, c))
      os << '\\';
    os << c;
  }
  return buf;
}

using AttributeMap = std::map<std::string, std::string>;

namespace {

/// This struct represents a node in the DOT language. Each node has an
/// identifier and an optional identifier for the cluster (subgraph) that
/// contains the node.
/// Note: In the DOT language, edges can be drawn only from nodes to nodes, but
/// not between clusters. However, edges can be clipped to the boundary of a
/// cluster with `lhead` and `ltail` attributes. Therefore, when creating a new
/// cluster, an invisible "anchor" node is created.
struct Node {
public:
  Node(int id = 0, std::optional<int> clusterId = std::nullopt)
      : id(id), clusterId(clusterId) {}

  int id;
  std::optional<int> clusterId;
};

struct DataFlowEdge {
  Value value;
  Node node;
  /// Port on the source node (the `<res...>` port of the value's producer, or
  /// the `<arg...>` port of an input bar). Identifies the value.
  std::string srcPort;
  /// Port on the consuming node. Identifies the specific operand slot, so that
  /// passing the same value to several operands yields distinct ports.
  std::string dstPort;
  /// If set, the edge originates from this node instead of the node that
  /// produces `value`. Used to route uses of a region op's operand through the
  /// op's input bar.
  std::optional<Node> source = std::nullopt;
  /// When true, the source attaches to its `<arg...>` port (south side) rather
  /// than its `<res...>` port. Used when `source` is an input bar.
  bool sourceUsesArgPort = false;
};

/// The anchor node of a block's cluster and the node of its last (terminator)
/// operation. Used to bracket a region op's input/output bars: the input bar is
/// ordered above the anchor (top) and the output bar below the terminator
/// (bottom).
struct BlockNodes {
  Node anchor;
  Node last;
};

/// An input bar node together with the `<arg...>` port that a particular value
/// occupies on it.
using InputBarPort = std::pair<Node, std::string>;

/// This pass generates a Graphviz dataflow visualization of an MLIR operation.
/// Note: See https://www.graphviz.org/doc/info/lang.html for more information
/// about the Graphviz DOT language.
class PrintOpPass : public impl::ViewOpGraphPassBase<PrintOpPass> {
public:
  PrintOpPass() : os(llvm::errs()) {}
  explicit PrintOpPass(ViewOpGraphPassOptions options)
      : impl::ViewOpGraphPassBase<PrintOpPass>(std::move(options)),
        os(llvm::errs()) {}
  PrintOpPass(raw_ostream &os) : os(os) {}
  PrintOpPass(const PrintOpPass &o) : PrintOpPass(o.os.getOStream()) {}

  void runOnOperation() override {
    initColorMapping(*getOperation());
    emitGraph([&]() {
      processOperation(getOperation());
      emitAllEdgeStmts();
    });
    markAllAnalysesPreserved();
  }

  /// Create a CFG graph for a region. Used in `Region::viewGraph`.
  void emitRegionCFG(Region &region) {
    printControlFlowEdges = true;
    printDataFlowEdges = false;
    initColorMapping(region);
    emitGraph([&]() { processRegion(region); });
  }

private:
  /// Generate a color mapping that will color every operation with the same
  /// name the same way. It'll interpolate the hue in the HSV color-space,
  /// using muted colors that provide good contrast for black text.
  template <typename T>
  void initColorMapping(T &irEntity) {
    backgroundColors.clear();
    SmallVector<Operation *> ops;
    irEntity.walk([&](Operation *op) {
      auto &entry = backgroundColors[op->getName()];
      if (entry.first == 0)
        ops.push_back(op);
      ++entry.first;
    });
    for (auto indexedOps : llvm::enumerate(ops)) {
      double hue = ((double)indexedOps.index()) / ops.size();
      // Use lower saturation (0.3) and higher value (0.95) for better
      // readability
      backgroundColors[indexedOps.value()->getName()].second =
          std::to_string(hue) + " 0.3 0.95";
    }
  }

  /// Emit all edges. This function should be called after all nodes have been
  /// emitted.
  void emitAllEdgeStmts() {
    if (printDataFlowEdges) {
      for (const auto &e : dataFlowEdges) {
        Node source = e.source ? *e.source : valueToNode[e.value];
        emitEdgeStmt(source, e.node, e.srcPort, e.dstPort, kLineStyleDataFlow,
                     e.sourceUsesArgPort);
      }
    }

    for (const std::string &edge : edges)
      os << edge << ";\n";
    edges.clear();
  }

  /// Emit a cluster (subgraph). The specified builder generates the body of the
  /// cluster. Return the anchor node of the cluster.
  Node emitClusterStmt(function_ref<void()> builder,
                       const std::string &label = "") {
    int clusterId = ++counter;
    os << "subgraph cluster_" << clusterId << " {\n";
    os.indent();
    // Emit invisible anchor node from/to which arrows can be drawn.
    Node anchorNode = emitNodeStmt(" ", kShapeNone);
    os << attrStmt("label", quoteString(label)) << ";\n";
    builder();
    os.unindent();
    os << "}\n";
    return Node(anchorNode.id, clusterId);
  }

  /// Generate an attribute statement.
  std::string attrStmt(const Twine &key, const Twine &value) {
    return (key + " = " + value).str();
  }

  /// Emit an attribute list.
  void emitAttrList(raw_ostream &os, const AttributeMap &map) {
    os << "[";
    interleaveComma(map, os, [&](const auto &it) {
      os << this->attrStmt(it.first, it.second);
    });
    os << "]";
  }

  // Print an MLIR attribute to `os`. Large attributes are truncated.
  void emitMlirAttr(raw_ostream &os, Attribute attr) {
    // A value used to elide large container attribute.
    int64_t largeAttrLimit = getLargeAttributeSizeLimit();

    // Always emit splat attributes.
    if (isa<SplatElementsAttr>(attr)) {
      os << escapeLabelString(
          strFromOs([&](raw_ostream &os) { attr.print(os); }));
      return;
    }

    // Elide "big" elements attributes.
    auto elements = dyn_cast<ElementsAttr>(attr);
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getShapedType().getRank(), '[') << "..."
         << std::string(elements.getShapedType().getRank(), ']') << " : ";
      emitMlirType(os, elements.getType());
      return;
    }

    auto array = dyn_cast<ArrayAttr>(attr);
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      return;
    }

    // Print all other attributes.
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    attr.print(ss);
    os << escapeLabelString(truncateString(buf));
  }

  // Print a truncated and escaped MLIR type to `os`.
  void emitMlirType(raw_ostream &os, Type type) {
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    type.print(ss);
    os << escapeLabelString(truncateString(buf));
  }

  // Print a truncated and escaped MLIR operand to `os`.
  void emitMlirOperand(raw_ostream &os, Value operand) {
    operand.printAsOperand(os, OpPrintingFlags());
  }

  /// Append an edge to the list of edges.
  /// Note: Edges are written to the output stream via `emitAllEdgeStmts`.
  /// The edge leaves `n1` at `srcPort` and enters `n2` at `dstPort`. The source
  /// and destination ports differ when the same value feeds several operands,
  /// since each operand has its own (unique) input port. When
  /// `sourceUsesArgPort` is true, the source attaches at its `<arg...>` port
  /// (used when the source is a region op's input bar passing a value down into
  /// the region) instead of its `<res...>` port.
  void emitEdgeStmt(Node n1, Node n2, StringRef srcPort, StringRef dstPort,
                    StringRef style, bool sourceUsesArgPort = false) {
    AttributeMap attrs;
    attrs["style"] = style.str();
    // Use `ltail` and `lhead` to draw edges between clusters.
    if (n1.clusterId)
      attrs["ltail"] = "cluster_" + std::to_string(*n1.clusterId);
    if (n2.clusterId)
      attrs["lhead"] = "cluster_" + std::to_string(*n2.clusterId);

    edges.push_back(strFromOs([&](raw_ostream &os) {
      os << "v" << n1.id;
      if (!srcPort.empty() && !n1.clusterId) {
        // Attach edge to the south compass point of the source port. An input
        // bar exposes the value on its `<arg...>` port; every other node
        // produces it on its `<res...>` port.
        os << (sourceUsesArgPort ? ":arg" : ":res") << srcPort << ":s";
      }
      os << " -> ";
      os << "v" << n2.id;
      if (!dstPort.empty() && !n2.clusterId)
        // Attach edge to north compass point of the operand
        os << ":arg" << dstPort << ":n";
      emitAttrList(os, attrs);
    }));
  }

  /// Emit an invisible edge used only to constrain the relative rank (vertical
  /// position) of two nodes without drawing anything.
  void emitOrderingEdge(Node from, Node to) {
    AttributeMap attrs;
    attrs["style"] = "invis";
    os << "v" << from.id << " -> v" << to.id;
    emitAttrList(os, attrs);
    os << ";\n";
  }

  /// Emit a graph. The specified builder generates the body of the graph.
  void emitGraph(function_ref<void()> builder) {
    os << "digraph G {\n";
    os.indent();
    // Edges between clusters are allowed only in compound mode.
    os << attrStmt("compound", "true") << ";\n";
    builder();
    os.unindent();
    os << "}\n";
  }

  /// Emit a node statement.
  Node emitNodeStmt(const std::string &label, StringRef shape = kShapeNode,
                    StringRef background = "") {
    int nodeId = ++counter;
    AttributeMap attrs;
    attrs["label"] = quoteString(label);
    attrs["shape"] = shape.str();
    if (!background.empty()) {
      attrs["style"] = "filled";
      attrs["fillcolor"] = quoteString(background.str());
    }
    os << llvm::format("v%i ", nodeId);
    emitAttrList(os, attrs);
    os << ";\n";
    return Node(nodeId);
  }

  std::string getValuePortName(Value operand) {
    // Print value as an operand and omit the leading '%' character.
    auto str = strFromOs([&](raw_ostream &os) {
      operand.printAsOperand(os, OpPrintingFlags());
    });
    // Replace % and # with _
    llvm::replace(str, '%', '_');
    llvm::replace(str, '#', '_');
    return str;
  }

  /// Port name for the operand at `index`. Normally the value's name, but when
  /// the same value is passed to several operands it is suffixed with the
  /// operand index so each operand slot gets a unique input port (otherwise all
  /// edges would attach to the first such port).
  std::string getOperandPortName(Operation *op, unsigned index) {
    Value operand = op->getOperand(index);
    std::string name = getValuePortName(operand);
    if (llvm::count(op->getOperands(), operand) > 1)
      name += "_" + std::to_string(index);
    return name;
  }

  /// Generate the label of a region op's cluster: the op name followed by its
  /// attributes. Each line ends with `\l` so that, together with the cluster's
  /// `labeljust = "l"` attribute, the label is anchored in the top-left corner.
  /// Result types are intentionally omitted here since they are shown on the
  /// output bar's ports.
  std::string getClusterLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      os << op->getName() << "\\l";
      if (printAttrs) {
        for (const NamedAttribute &attr : op->getAttrs()) {
          os << escapeLabelString(attr.getName().getValue().str()) << ": ";
          emitMlirAttr(os, attr.getValue());
          os << "\\l";
        }
      }
    });
  }

  /// Emit the operand fields of a record label as a single horizontal row of
  /// input ports, e.g. `<arg_0> %a | <arg_1> %b`.
  void emitOperandPorts(raw_ostream &os, Operation *op) {
    interleave(
        llvm::enumerate(op->getOperands()), os,
        [&](auto operand) {
          os << "<arg" << getOperandPortName(op, operand.index()) << "> ";
          emitMlirOperand(os, operand.value());
        },
        "|");
  }

  /// Emit the result fields of a record label as a single horizontal row of
  /// output ports, e.g. `<res_0> %0 i32 | <res_1> %1 i64`.
  void emitResultPorts(raw_ostream &os, Operation *op) {
    interleave(
        op->getResults(), os,
        [&](Value result) {
          os << "<res" << getValuePortName(result) << "> ";
          emitMlirOperand(os, result);
          if (printResultTypes) {
            os << " ";
            emitMlirType(os, result.getType());
          }
        },
        "|");
  }

  /// Generate a label for an operation.
  std::string getRecordLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      os << "{";

      // Print operation inputs.
      if (op->getNumOperands() > 0) {
        os << "{";
        emitOperandPorts(os, op);
        os << "}|";
      }
      // Print operation name and type.
      os << op->getName() << "\\l";

      // Print attributes.
      if (printAttrs && !op->getAttrs().empty()) {
        // Extra line break to separate attributes from the operation name.
        os << "\\l";
        for (const NamedAttribute &attr : op->getAttrs()) {
          os << attr.getName().getValue() << ": ";
          emitMlirAttr(os, attr.getValue());
          os << "\\l";
        }
      }

      if (op->getNumResults() > 0) {
        os << "|{";
        emitResultPorts(os, op);
        os << "}";
      }

      os << "}";
    });
  }

  /// Generate the label for the "input bar" of a region op: an "operands" label
  /// followed by the operand ports, rendered as a horizontal bar near the top
  /// of the region's cluster.
  std::string getInputBarLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      os << "operands|";
      emitOperandPorts(os, op);
    });
  }

  /// Generate the label for the "output bar" of a region op: a "results" label
  /// followed by the result ports, rendered as a horizontal bar near the bottom
  /// of the region's cluster.
  std::string getOutputBarLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      os << "results|";
      emitResultPorts(os, op);
    });
  }

  /// Generate a label for a block argument.
  std::string getLabel(BlockArgument arg) {
    return strFromOs([&](raw_ostream &os) {
      os << "<res" << getValuePortName(arg) << "> ";
      arg.printAsOperand(os, OpPrintingFlags());
      if (printResultTypes) {
        os << " ";
        emitMlirType(os, arg.getType());
      }
    });
  }

  /// Process a block. Emit a cluster and one node per block argument and
  /// operation inside the cluster. Return the cluster's anchor node and the node
  /// of the block's last operation (its terminator); the latter falls back to
  /// the anchor for an empty block.
  BlockNodes processBlock(Block &block) {
    Node lastNode;
    bool hasLast = false;
    Node anchor = emitClusterStmt([&]() {
      for (BlockArgument &blockArg : block.getArguments())
        valueToNode[blockArg] = emitNodeStmt(getLabel(blockArg));
      // Emit a node for each operation.
      std::optional<Node> prevNode;
      for (Operation &op : block) {
        Node nextNode = processOperation(&op);
        if (printControlFlowEdges && prevNode)
          emitEdgeStmt(*prevNode, nextNode, /*srcPort=*/"", /*dstPort=*/"",
                       kLineStyleControlFlow);
        prevNode = nextNode;
        lastNode = nextNode;
        hasLast = true;
      }
    });
    return {anchor, hasLast ? lastNode : anchor};
  }

  /// Emit a cluster for an operation that has regions. The op's operands are
  /// shown as input ports on a record node ("input bar") near the top of the
  /// cluster, and its results as output ports on a record node ("output bar")
  /// near the bottom. The op title (name and attributes) is the cluster label,
  /// anchored top-left.
  ///
  /// Two invisible edges per block (input bar -> block anchor, and the block's
  /// terminator -> output bar) gently keep the region body between the bars, so
  /// data flow tends to enter from the top and leave from the bottom,
  /// preserving the top-to-bottom hierarchy of the graph. Pinning the output
  /// bar below the terminator (rather than below the block anchor) keeps it at
  /// the bottom edge even when the op has multiple regions. This is a soft
  /// nudge, not strict pinning; the rest of the layout is left to GraphViz.
  ///
  /// Uses of an operand inside the region are routed through the input bar (see
  /// `valueToInputBar`), so a value crosses the region boundary exactly once.
  ///
  /// Return a pair {input, output}: the node that consumes the op's operands
  /// (top) and the node that produces the op's results (bottom). When the op
  /// has no operands/results the corresponding node is an invisible anchor.
  std::pair<Node, Node> emitRegionOpCluster(Operation *op) {
    int clusterId = ++counter;
    os << "subgraph cluster_" << clusterId << " {\n";
    os.indent();
    // Anchor the op title in the top-left corner of the region's box.
    os << attrStmt("labeljust", quoteString("l")) << ";\n";
    os << attrStmt("labelloc", quoteString("t")) << ";\n";
    os << attrStmt("label", quoteString(getClusterLabel(op))) << ";\n";

    // Input bar: operand ports near the top. Use an invisible anchor when the
    // op has no operands so the cluster still has a representative node.
    Node inputNode =
        op->getNumOperands() > 0
            ? emitNodeStmt(getInputBarLabel(op), kShapeNode,
                           backgroundColors[op->getName()].second)
            : emitNodeStmt(" ", kShapeNone);

    // While emitting the region bodies, route uses of the op's operands through
    // the input bar (recording the bar node and the value's port on it). Save
    // and restore any outer mapping so nested region ops behave correctly.
    SmallVector<std::pair<Value, std::optional<InputBarPort>>> savedInputBars;
    for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
      Value operand = op->getOperand(i);
      auto it = valueToInputBar.find(operand);
      savedInputBars.emplace_back(
          operand, it != valueToInputBar.end()
                       ? std::optional<InputBarPort>(it->second)
                       : std::nullopt);
      valueToInputBar[operand] = {inputNode, getOperandPortName(op, i)};
    }

    // Emit the region bodies and collect the per-block nodes so they can be
    // ordered vertically between the input and output bars.
    SmallVector<BlockNodes> blocks;
    for (Region &region : op->getRegions())
      llvm::append_range(blocks, processRegion(region));

    for (auto &[operand, prev] : savedInputBars) {
      if (prev)
        valueToInputBar[operand] = *prev;
      else
        valueToInputBar.erase(operand);
    }

    // Output bar: result ports near the bottom, or an invisible anchor.
    Node outputNode =
        op->getNumResults() > 0
            ? emitNodeStmt(getOutputBarLabel(op), kShapeNode,
                           backgroundColors[op->getName()].second)
            : emitNodeStmt(" ", kShapeNone);

    // Softly keep the input bar above each block (via its anchor) and the
    // output bar below each block (via its terminator) using invisible edges.
    if (blocks.empty()) {
      emitOrderingEdge(inputNode, outputNode);
    } else {
      for (const BlockNodes &block : blocks) {
        emitOrderingEdge(inputNode, block.anchor);
        emitOrderingEdge(block.last, outputNode);
      }
    }

    os.unindent();
    os << "}\n";
    return {inputNode, outputNode};
  }

  /// Process an operation. If the operation has regions, emit a cluster.
  /// Otherwise, emit a node.
  Node processOperation(Operation *op) {
    // `sink` consumes the op's operands (top of the node/cluster); `source`
    // produces the op's results (bottom). For ops without regions these are the
    // same record node.
    Node sink, source;
    if (op->getNumRegions() > 0) {
      std::pair<Node, Node> anchors = emitRegionOpCluster(op);
      sink = anchors.first;
      source = anchors.second;
    } else {
      Node node = emitNodeStmt(getRecordLabel(op), kShapeNode,
                               backgroundColors[op->getName()].second);
      sink = source = node;
    }

    // Insert data flow edges originating from each operand. If the operand is an
    // operand of an enclosing region op, route the edge from that op's input
    // bar instead of directly from the value's producer.
    if (printDataFlowEdges) {
      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++) {
        Value operand = op->getOperand(i);
        std::string dstPort = getOperandPortName(op, i);
        auto it = valueToInputBar.find(operand);
        if (it != valueToInputBar.end())
          // Route through the enclosing region op's input bar: source is that
          // bar, using its `<arg...>` port for the value.
          dataFlowEdges.push_back({operand, sink, it->second.second, dstPort,
                                   it->second.first,
                                   /*sourceUsesArgPort=*/true});
        else
          dataFlowEdges.push_back(
              {operand, sink, getValuePortName(operand), dstPort});
      }
    }

    for (Value result : op->getResults())
      valueToNode[result] = source;

    return sink;
  }

  /// Process a region. Return the anchor and terminator node of each contained
  /// block.
  SmallVector<BlockNodes> processRegion(Region &region) {
    SmallVector<BlockNodes> blocks;
    for (Block &block : region.getBlocks())
      blocks.push_back(processBlock(block));
    return blocks;
  }

  /// Truncate long strings.
  std::string truncateString(std::string str) {
    if (str.length() <= maxLabelLen)
      return str;
    return str.substr(0, maxLabelLen) + "...";
  }

  /// Output stream to write DOT file to.
  raw_indented_ostream os;
  /// A list of edges. For simplicity, should be emitted after all nodes were
  /// emitted.
  std::vector<std::string> edges;
  /// Mapping of SSA values to Graphviz nodes/clusters.
  DenseMap<Value, Node> valueToNode;
  /// Mapping of a region op's operand values to the op's input bar node and the
  /// value's port on it, active only while the op's regions are being emitted.
  /// Uses of these values inside the region are routed through the input bar.
  DenseMap<Value, InputBarPort> valueToInputBar;
  /// Output for data flow edges is delayed until the end to handle cycles
  std::vector<DataFlowEdge> dataFlowEdges;
  /// Counter for generating unique node/subgraph identifiers.
  int counter = 0;

  DenseMap<OperationName, std::pair<int, std::string>> backgroundColors;
};

} // namespace

std::unique_ptr<Pass> mlir::createViewOpGraphPass(raw_ostream &os) {
  return std::make_unique<PrintOpPass>(os);
}

/// Generate a CFG for a region and show it in a window.
static void llvmViewGraph(Region &region, const Twine &name) {
  int fd;
  std::string filename = llvm::createGraphFilename(name.str(), fd);
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    if (fd == -1) {
      llvm::errs() << "error opening file '" << filename << "' for writing\n";
      return;
    }
    PrintOpPass pass(os);
    pass.emitRegionCFG(region);
  }
  llvm::DisplayGraph(filename, /*wait=*/false, llvm::GraphProgram::DOT);
}

void mlir::Region::viewGraph(const Twine &regionName) {
  llvmViewGraph(*this, regionName);
}

void mlir::Region::viewGraph() { viewGraph("region"); }
