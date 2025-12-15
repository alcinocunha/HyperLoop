# HyperLasso: a Bounded Model Checker for HyperLTL

## Description

HyperLasso considers only infinite traces that can be represented in the lasso form: a finite prefix followed by an infinitely repeated lasso. This is the same semantics that complete (non-bounded) model checkers assume for normal LTL.

The user specifies the exact length of the traces to be checked. Since we assume traces to be infinite, for a given length all shorter lasso traces are also considered, so there is no need to iterate the length for completeness.

HyperLasso is implemented in Python and translates the model and the HyperLTL property into a set of constraints using quantifiers that are solved using the Z3 SMT solver. Unlike other existing bounded model checkers for HyperLTL (namely [HyperQB](https://github.com/HyperQB/HyperRUSTY/)), HyperLasso fully supports loop conditions so the results are complete for the given trace length.

Models should be defined in the [SMV language](https://nusmv.fbk.eu) and the HyperLTL property should be specified in a separate file following the HyperQB syntax. For the moment, HyperLasso only support declarative models defined using the `INIT`, `TRANS`, and `INVAR` sections. For most examples, this actually results in more concise models than using explicit models defined with the `ASSIGN` section.

## Usage

To run HyperLasso, you must first install Z3 with `pip install z3-solver` and then execute the following command:
```bash
python HyperLasso.py <property_file.hq> <trace_length> <model_file1.smv> [<model_file2.smv> ...]
```

You can give one SMV model for quantified trace in the property or a single model for all traces. `<trace_length>` is the exact length of the traces that will be checked.

See example SMV files and the respective properties in the `examples/` directory.