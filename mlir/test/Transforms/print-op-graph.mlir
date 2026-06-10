// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -view-op-graph %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s
// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -view-op-graph='print-data-flow-edges=false print-control-flow-edges=true' %s -o %t 2>&1 | FileCheck -check-prefix=CFG %s

// DFG-LABEL: digraph G {
//  DFG-NEXT:   compound = true;
//  DFG-NEXT:   subgraph cluster_1 {
//  DFG-NEXT:     labeljust = "l";
//  DFG-NEXT:     labelloc = "t";
//  DFG-NEXT:     label = "builtin.module\l";
//  DFG-NEXT:     v2 [label = " ", shape = plain];
//  DFG-NEXT:     subgraph cluster_3 {
//  DFG-NEXT:       v4 [label = " ", shape = plain];
//  DFG-NEXT:       label = "";
//  DFG-NEXT:       subgraph cluster_5 {
//  DFG-NEXT:         labeljust = "l";
//  DFG-NEXT:         labelloc = "t";
//  DFG-NEXT:         label = "func.func\lfunction_type: (i32, i32) -\> (i32, ...\lsym_name: \"merge_blocks\"\l";
//  DFG-NEXT:         v6 [label = " ", shape = plain];
//  DFG-NEXT:         subgraph cluster_7 {
//  DFG-NEXT:           v8 [label = " ", shape = plain];
//  DFG-NEXT:           label = "";
//  DFG-NEXT:           v9 [label = "<res_arg0> %arg0 i32", shape = Mrecord];
//  DFG-NEXT:           v10 [label = "<res_arg1> %arg1 i32", shape = Mrecord];
//  DFG-NEXT:           v11 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: {{\[\[}}...]] : tensor\<2x2xi32\>\l|{<res_cst> %cst tensor\<2x2xi32\>}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           v12 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: dense\<1\> : tensor\<5xi32\>\l|{<res_cst_0> %cst_0 tensor\<5xi32\>}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           v13 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: dense\<{{\[\[}}0, 1]]\> : te...\l|{<res_cst_1> %cst_1 tensor\<1x2xi32\>}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           v14 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: 10 : i32\l|{<res_c10_i32> %c10_i32 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           v15 [fillcolor = "0.142857 0.3 0.95", label = "{test.func\l|{<res_0> %0 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           subgraph cluster_16 {
//  DFG-NEXT:             labeljust = "l";
//  DFG-NEXT:             labelloc = "t";
//  DFG-NEXT:             label = "test.merge_blocks\l";
//  DFG-NEXT:             v17 [label = " ", shape = plain];
//  DFG-NEXT:             subgraph cluster_18 {
//  DFG-NEXT:               v19 [label = " ", shape = plain];
//  DFG-NEXT:               label = "";
//  DFG-NEXT:               v20 [fillcolor = "0.285714 0.3 0.95", label = "{{\{\{}}<arg_arg0> %arg0|<arg_0> %0|<arg_c10_i32> %c10_i32}|test.br\l}", shape = Mrecord, style = filled];
//  DFG-NEXT:             }
//  DFG-NEXT:             subgraph cluster_21 {
//  DFG-NEXT:               v22 [label = " ", shape = plain];
//  DFG-NEXT:               label = "";
//  DFG-NEXT:               v23 [label = "<res_2> %2 i32", shape = Mrecord];
//  DFG-NEXT:               v24 [label = "<res_3> %3 i32", shape = Mrecord];
//  DFG-NEXT:               v25 [label = "<res_4> %4 i32", shape = Mrecord];
//  DFG-NEXT:               v26 [fillcolor = "0.428571 0.3 0.95", label = "{{\{\{}}<arg_2> %2|<arg_3> %3}|test.return\l}", shape = Mrecord, style = filled];
//  DFG-NEXT:             }
//  DFG-NEXT:             v27 [fillcolor = "0.571429 0.3 0.95", label = "results|<res_1_0> %1#0 i32|<res_1_1> %1#1 i32", shape = Mrecord, style = filled];
//  DFG-NEXT:             v17 -> v19[style = invis];
//  DFG-NEXT:             v20 -> v27[style = invis];
//  DFG-NEXT:             v17 -> v22[style = invis];
//  DFG-NEXT:             v26 -> v27[style = invis];
//  DFG-NEXT:           }
//  DFG-NEXT:           v28 [fillcolor = "0.428571 0.3 0.95", label = "{{\{\{}}<arg_1_0> %1#0|<arg_1_1> %1#1}|test.return\l}", shape = Mrecord, style = filled];
//  DFG-NEXT:         }
//  DFG-NEXT:         v29 [label = " ", shape = plain];
//  DFG-NEXT:         v6 -> v8[style = invis];
//  DFG-NEXT:         v28 -> v29[style = invis];
//  DFG-NEXT:       }
//  DFG-NEXT:     }
//  DFG-NEXT:     v30 [label = " ", shape = plain];
//  DFG-NEXT:     v2 -> v4[style = invis];
//  DFG-NEXT:     v6 -> v30[style = invis];
//  DFG-NEXT:   }
//  DFG-NEXT:   v9:res_arg0:s -> v20:arg_arg0:n[style = solid];
//  DFG-NEXT:   v15:res_0:s -> v20:arg_0:n[style = solid];
//  DFG-NEXT:   v14:res_c10_i32:s -> v20:arg_c10_i32:n[style = solid];
//  DFG-NEXT:   v23:res_2:s -> v26:arg_2:n[style = solid];
//  DFG-NEXT:   v24:res_3:s -> v26:arg_3:n[style = solid];
//  DFG-NEXT:   v27:res_1_0:s -> v28:arg_1_0:n[style = solid];
//  DFG-NEXT:   v27:res_1_1:s -> v28:arg_1_1:n[style = solid];
//  DFG-NEXT: }

// CFG-LABEL: digraph G {
//  CFG-NEXT:   compound = true;
//  CFG-NEXT:   subgraph cluster_1 {
//  CFG-NEXT:     labeljust = "l";
//  CFG-NEXT:     labelloc = "t";
//  CFG-NEXT:     label = "builtin.module\l";
//  CFG-NEXT:     v2 [label = " ", shape = plain];
//  CFG-NEXT:     subgraph cluster_3 {
//  CFG-NEXT:       v4 [label = " ", shape = plain];
//  CFG-NEXT:       label = "";
//  CFG-NEXT:       subgraph cluster_5 {
//  CFG-NEXT:         labeljust = "l";
//  CFG-NEXT:         labelloc = "t";
//  CFG-NEXT:         label = "func.func\lfunction_type: (i32, i32) -\> (i32, ...\lsym_name: \"merge_blocks\"\l";
//  CFG-NEXT:         v6 [label = " ", shape = plain];
//  CFG-NEXT:         subgraph cluster_7 {
//  CFG-NEXT:           v8 [label = " ", shape = plain];
//  CFG-NEXT:           label = "";
//  CFG-NEXT:           v9 [label = "<res_arg0> %arg0 i32", shape = Mrecord];
//  CFG-NEXT:           v10 [label = "<res_arg1> %arg1 i32", shape = Mrecord];
//  CFG-NEXT:           v11 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: {{\[\[}}...]] : tensor\<2x2xi32\>\l|{<res_cst> %cst tensor\<2x2xi32\>}}", shape = Mrecord, style = filled];
//  CFG-NEXT:           v12 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: dense\<1\> : tensor\<5xi32\>\l|{<res_cst_0> %cst_0 tensor\<5xi32\>}}", shape = Mrecord, style = filled];
//  CFG-NEXT:           v13 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: dense\<{{\[\[}}0, 1]]\> : te...\l|{<res_cst_1> %cst_1 tensor\<1x2xi32\>}}", shape = Mrecord, style = filled];
//  CFG-NEXT:           v14 [fillcolor = "0.000000 0.3 0.95", label = "{arith.constant\l\lvalue: 10 : i32\l|{<res_c10_i32> %c10_i32 i32}}", shape = Mrecord, style = filled];
//  CFG-NEXT:           v15 [fillcolor = "0.142857 0.3 0.95", label = "{test.func\l|{<res_0> %0 i32}}", shape = Mrecord, style = filled];
//  CFG-NEXT:           subgraph cluster_16 {
//  CFG-NEXT:             labeljust = "l";
//  CFG-NEXT:             labelloc = "t";
//  CFG-NEXT:             label = "test.merge_blocks\l";
//  CFG-NEXT:             v17 [label = " ", shape = plain];
//  CFG-NEXT:             subgraph cluster_18 {
//  CFG-NEXT:               v19 [label = " ", shape = plain];
//  CFG-NEXT:               label = "";
//  CFG-NEXT:               v20 [fillcolor = "0.285714 0.3 0.95", label = "{{\{\{}}<arg_arg0> %arg0|<arg_0> %0|<arg_c10_i32> %c10_i32}|test.br\l}", shape = Mrecord, style = filled];
//  CFG-NEXT:             }
//  CFG-NEXT:             subgraph cluster_21 {
//  CFG-NEXT:               v22 [label = " ", shape = plain];
//  CFG-NEXT:               label = "";
//  CFG-NEXT:               v23 [label = "<res_2> %2 i32", shape = Mrecord];
//  CFG-NEXT:               v24 [label = "<res_3> %3 i32", shape = Mrecord];
//  CFG-NEXT:               v25 [label = "<res_4> %4 i32", shape = Mrecord];
//  CFG-NEXT:               v26 [fillcolor = "0.428571 0.3 0.95", label = "{{\{\{}}<arg_2> %2|<arg_3> %3}|test.return\l}", shape = Mrecord, style = filled];
//  CFG-NEXT:             }
//  CFG-NEXT:             v27 [fillcolor = "0.571429 0.3 0.95", label = "results|<res_1_0> %1#0 i32|<res_1_1> %1#1 i32", shape = Mrecord, style = filled];
//  CFG-NEXT:             v17 -> v19[style = invis];
//  CFG-NEXT:             v20 -> v27[style = invis];
//  CFG-NEXT:             v17 -> v22[style = invis];
//  CFG-NEXT:             v26 -> v27[style = invis];
//  CFG-NEXT:           }
//  CFG-NEXT:           v28 [fillcolor = "0.428571 0.3 0.95", label = "{{\{\{}}<arg_1_0> %1#0|<arg_1_1> %1#1}|test.return\l}", shape = Mrecord, style = filled];
//  CFG-NEXT:         }
//  CFG-NEXT:         v29 [label = " ", shape = plain];
//  CFG-NEXT:         v6 -> v8[style = invis];
//  CFG-NEXT:         v28 -> v29[style = invis];
//  CFG-NEXT:       }
//  CFG-NEXT:     }
//  CFG-NEXT:     v30 [label = " ", shape = plain];
//  CFG-NEXT:     v2 -> v4[style = invis];
//  CFG-NEXT:     v6 -> v30[style = invis];
//  CFG-NEXT:   }
//  CFG-NEXT:   v11 -> v12[style = dashed];
//  CFG-NEXT:   v12 -> v13[style = dashed];
//  CFG-NEXT:   v13 -> v14[style = dashed];
//  CFG-NEXT:   v14 -> v15[style = dashed];
//  CFG-NEXT:   v15 -> v17[style = dashed];
//  CFG-NEXT:   v17 -> v28[style = dashed];
//  CFG-NEXT: }

func.func @merge_blocks(%arg0: i32, %arg1 : i32) -> (i32, i32) {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = arith.constant dense<1> : tensor<5xi32>
  %2 = arith.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %a = arith.constant 10 : i32
  %b = "test.func"() : () -> i32
  %3:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %b, %a)[^bb1] : (i32, i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32, %arg5: i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%3#0, %3#1) : (i32, i32) -> ()
}
