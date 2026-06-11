// RUN: mlir-opt -allow-unregistered-dialect -view-op-graph %s -o %t 2>&1 | FileCheck %s

// Regression test: when the same value is passed to several operands, each
// operand must get its own input port (<arg_0_0>, <arg_0_1>, ...) so that the
// data flow edges attach to distinct ports instead of all landing on the first.

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
//  CHECK-NEXT:       v6 [fillcolor = "0.333333 0.3 0.95", label = "{{\{\{}}<arg_0_0> %0|<arg_0_1> %0}|test.add\l}", shape = Mrecord, style = filled];
//  CHECK-NEXT:     }
//  CHECK-NEXT:     v7 [label = " ", shape = plain];
//  CHECK-NEXT:     v2 -> v4[style = invis];
//  CHECK-NEXT:     v6 -> v7[style = invis];
//  CHECK-NEXT:   }
//  CHECK-NEXT:   v5:res_0:s -> v6:arg_0_0:n[style = solid];
//  CHECK-NEXT:   v5:res_0:s -> v6:arg_0_1:n[style = solid];
//  CHECK-NEXT: }

%a = "test.producer"() : () -> i32
"test.add"(%a, %a) : (i32, i32) -> ()
