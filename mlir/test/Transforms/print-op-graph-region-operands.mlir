// RUN: mlir-opt -allow-unregistered-dialect -view-op-graph %s -o %t 2>&1 | FileCheck %s

// Regression test: a region op (test.region_op) with operands. Its operands
// are drawn as an input "bar" (an "operands" record of <arg_*> ports) near the
// top of the region cluster, and its result as an output bar near the bottom. A
// use of an operand inside the region (%0 in test.inner) is routed through the
// input bar (v8:arg_0:s -> ...) rather than directly from the external producer,
// so the value crosses the region boundary exactly once.

// CHECK-LABEL: digraph G {
//  CHECK-NEXT:   compound = true;
//  CHECK-NEXT:   subgraph cluster_1 {
//  CHECK-NEXT:     labeljust = "l";
//  CHECK-NEXT:     labelloc = "t";
//  CHECK-NEXT:     label = "builtin.module\l";
//  CHECK-NEXT:     v2 [label = " ", shape = plain];
//  CHECK-NEXT:     subgraph cluster_3 {
//  CHECK-NEXT:       v4 [label = " ", shape = plain];
//  CHECK-NEXT:       label = "";
//  CHECK-NEXT:       v5 [fillcolor = "0.000000 0.3 0.95", label = "{test.producer\l|{<res_0> %0 i32}}", shape = Mrecord, style = filled];
//  CHECK-NEXT:       v6 [fillcolor = "0.000000 0.3 0.95", label = "{test.producer\l|{<res_1> %1 i32}}", shape = Mrecord, style = filled];
//  CHECK-NEXT:       subgraph cluster_7 {
//  CHECK-NEXT:         labeljust = "l";
//  CHECK-NEXT:         labelloc = "t";
//  CHECK-NEXT:         label = "test.region_op\ltag: \"demo\"\l";
//  CHECK-NEXT:         v8 [fillcolor = "0.500000 0.3 0.95", label = "operands|<arg_0> %0|<arg_1> %1", shape = Mrecord, style = filled];
//  CHECK-NEXT:         subgraph cluster_9 {
//  CHECK-NEXT:           v10 [label = " ", shape = plain];
//  CHECK-NEXT:           label = "";
//  CHECK-NEXT:           v11 [fillcolor = "0.166667 0.3 0.95", label = "{{\{\{}}<arg_0> %0}|test.inner\l|{<res_3> %3 i32}}", shape = Mrecord, style = filled];
//  CHECK-NEXT:           v12 [fillcolor = "0.333333 0.3 0.95", label = "{{\{\{}}<arg_3> %3}|test.term\l}", shape = Mrecord, style = filled];
//  CHECK-NEXT:         }
//  CHECK-NEXT:         v13 [fillcolor = "0.500000 0.3 0.95", label = "results|<res_2> %2 i32", shape = Mrecord, style = filled];
//  CHECK-NEXT:         v8 -> v10[style = invis];
//  CHECK-NEXT:         v12 -> v13[style = invis];
//  CHECK-NEXT:       }
//  CHECK-NEXT:       v14 [fillcolor = "0.666667 0.3 0.95", label = "{{\{\{}}<arg_2> %2}|test.consumer\l}", shape = Mrecord, style = filled];
//  CHECK-NEXT:     }
//  CHECK-NEXT:     v15 [label = " ", shape = plain];
//  CHECK-NEXT:     v2 -> v4[style = invis];
//  CHECK-NEXT:     v14 -> v15[style = invis];
//  CHECK-NEXT:   }
//  CHECK-NEXT:   v8:arg_0:s -> v11:arg_0:n[style = solid];
//  CHECK-NEXT:   v11:res_3:s -> v12:arg_3:n[style = solid];
//  CHECK-NEXT:   v5:res_0:s -> v8:arg_0:n[style = solid];
//  CHECK-NEXT:   v6:res_1:s -> v8:arg_1:n[style = solid];
//  CHECK-NEXT:   v13:res_2:s -> v14:arg_2:n[style = solid];
//  CHECK-NEXT: }

%a = "test.producer"() : () -> i32
%b = "test.producer"() : () -> i32
%r = "test.region_op"(%a, %b) ({
  %inner = "test.inner"(%a) : (i32) -> i32
  "test.term"(%inner) : (i32) -> ()
}) {tag = "demo"} : (i32, i32) -> i32
"test.consumer"(%r) : (i32) -> ()
