"""
Bounded Model Checker for HyperLTL specifications over SMV models using Z3 for infinite lasso-shaped traces.
"""

from z3 import *
import sys
import itertools
from math import lcm
from Parser import *

# ===========================================================================
# Translate HyperLTL expressions to Z3
# ===========================================================================

def smv_temporal_expr_to_z3(K, i, expr: ASTNode, state: dict, loops: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        if expr.trace is not None:
            loop = loops[expr.trace]
            lasso = K - loop
            if i < loop:
                index = i
            else:
                index = loop + ((i - loop) % lasso)
            return state[expr.trace][index][expr.name]
        else:
            raise NotImplementedError("Non projected identifiers not supported in temporal formulas.")
    elif isinstance(expr, UnaryOp):
        if expr.operator == '!':
            operand = smv_temporal_expr_to_z3(K, i, expr.operand, state, loops)
            return Not(operand)
        elif expr.operator == '-':
            operand = smv_temporal_expr_to_z3(K, i, expr.operand, state, loops)
            return (- operand)
        elif expr.operator == 'G':
            lasso = { t: K - loops[t] for t in loops }
            loop = max(loops.values())
            length = loop + lcm(*lasso.values())
            return And([smv_temporal_expr_to_z3(K, j, expr.operand, state, loops) for j in range(min(i,loop), length)])
        elif expr.operator == 'F':
            lasso = { t: K - loops[t] for t in loops }
            loop = max(loops.values())
            length = loop + lcm(*lasso.values())
            return Or([smv_temporal_expr_to_z3(K, j, expr.operand, state, loops) for j in range(min(i,loop), length)])     
        elif expr.operator == 'X':
            lasso = { t: K - loops[t] for t in loops }
            loop = max(loops.values())
            length = loop + lcm(*lasso.values())
            if i + 1 < length:
                return smv_temporal_expr_to_z3(K, i + 1, expr.operand, state, loops)
            else:
                return smv_temporal_expr_to_z3(K, loop, expr.operand, state, loops)
    elif isinstance(expr, BinaryOp):
        left = smv_temporal_expr_to_z3(K, i, expr.left, state, loops)
        right = smv_temporal_expr_to_z3(K, i, expr.right, state, loops)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        raise NotImplementedError("Next not allowed in temporal formulas.")
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")


# ===========================================================================
# Translate SMV expressions to Z3
# ===========================================================================

def smv_expr_to_z3(expr: ASTNode, state: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        if expr.trace is not None:
            raise NotImplementedError("Projected identifiers not supported in state formulas.")
        return state[expr.name]
    elif isinstance(expr, UnaryOp):
        operand = smv_expr_to_z3(expr.operand, state)
        if expr.operator == '!':
            return Not(operand)
        elif expr.operator == '-':
            return (- operand)
        elif expr.operator == 'G':
            raise NotImplementedError("Temporal operator not allowed in state formulas.")
        elif expr.operator == 'F':
            raise NotImplementedError("Temporal operator not allowed in state formulas.")
    elif isinstance(expr, BinaryOp):
        left = smv_expr_to_z3(expr.left, state)
        right = smv_expr_to_z3(expr.right, state)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        raise NotImplementedError("Next not allowed in state formulas.")
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")

def smv_next_expr_to_z3(expr: ASTNode, state1: dict, state2: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        return state1[expr.name]
    elif isinstance(expr, UnaryOp):
        operand = smv_next_expr_to_z3(expr.operand, state1, state2)
        if expr.operator == '!':
            return Not(operand)
        elif expr.operator == '-':
            return (- operand)
    elif isinstance(expr, BinaryOp):
        left = smv_next_expr_to_z3(expr.left, state1, state2)
        right = smv_next_expr_to_z3(expr.right, state1, state2)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        return state2[expr.expr.name]
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")

# ============================================================================
# BMC procedure for HyperLTL
# ============================================================================

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) < 4:
        print("Usage: python HyperLasso.py <property_file.hq> <trace_length> <model_file1.smv> [<model_file2.smv> ...]")
        sys.exit(1)
    property_file = sys.argv[1]
    K = int(sys.argv[2])
    model_files = sys.argv[3:]

    with open(property_file, "r") as pf:
        formula = pf.read()
        hltl_spec = parse_hyperltl(formula)
        if len(model_files) == 1:
            modules = [parse_smv(open(model_files[0]).read())] * len(hltl_spec.vars)
        elif len(hltl_spec.vars) != len(model_files):
            print("Error: Number of model files must match number of quantified traces in the HyperLTL specification.")
            sys.exit(1)
        else:
            modules = [parse_smv(open(mf).read()) for mf in model_files]
        
    def frozen_vars(module,name):
        decls = {}
        for decl in module.frozenvar_decls:
            if decl.var_type.__class__ == RangeType:
                decls[decl.name] = Int(f"{name}_{decl.name}")
            elif decl.var_type.__class__ == BooleanType:
                decls[decl.name] = Bool(f"{name}_{decl.name}")
        return decls
    
    def state_vars(module,name,i):
        decls = {}
        for decl in module.var_decls:
            if decl.var_type.__class__ == RangeType:
                decls[decl.name] = Int(f"{name}_{i}_{decl.name}")
            elif decl.var_type.__class__ == BooleanType:
                decls[decl.name] = Bool(f"{name}_{i}_{decl.name}")
        return decls
    
    def init(module,s):
        if module.init_expr is None:
            return BoolVal(True)
        return smv_expr_to_z3(module.init_expr, s,)
    
    def trans(module,s1,s2):
        if module.trans_expr is None:
            return BoolVal(True)
        return smv_next_expr_to_z3(module.trans_expr, s1, s2)

    def invar(module,s):
        if module.invar_expr is None:
            return BoolVal(True)
        return smv_expr_to_z3(module.invar_expr, s)

    def behavior(module,state,loop,init,trans,invar):
        constraints = [loop >= 0, loop < K, init(module,state[0])]
        for decl in module.frozenvar_decls:
            if decl.var_type.__class__ == RangeType:
                var = state[0][decl.name]
                constraints.append(var >= decl.var_type.lower)
                constraints.append(var <= decl.var_type.upper)
        for i in range(K-1):
            constraints.append(trans(module,state[i], state[i+1]))
        for i in range(K):
            constraints.append(Implies(loop == i,trans(module,state[K-1], state[i])))
        for i in range(K):
            constraints.append(invar(module,state[i]))
        for i in range(K):
            for decl in module.var_decls:
                if decl.var_type.__class__ == RangeType:
                    var = state[i][decl.name]
                    constraints.append(var >= decl.var_type.lower)
                    constraints.append(var <= decl.var_type.upper)
        return And(constraints)
    
    def print_trace(model,state,loop,K):
        for i in range(K):
            print(f"State {i}:")
            for var in state[i]:
                print(f"  {var} = {model[state[i][var]]}")
        l = model[loop].as_long()
        print(f"Loop back to state: {l}")

    s = Solver()

    N = len(hltl_spec.vars)

    first_exists = N
    state = {}
    loop = {}
    for i,t in enumerate(hltl_spec.vars):
        if hltl_spec.quantifiers[i] == 'Exists':
            first_exists = min(first_exists, i)
        frozen = frozen_vars(modules[i],t)
        state[t] = [state_vars(modules[i], t, j) | frozen for j in range(K)]
        loop[t] = Int(f"loop_{t}")

    for i in range(first_exists):
        s.add(behavior(modules[i], state[hltl_spec.vars[i]], loop[hltl_spec.vars[i]], init, trans, invar))

    exprs = []
    for l in itertools.product(range(K), repeat=N):
        loops = {}
        for i,t in enumerate(hltl_spec.vars):
            loops[t] = l[i]    
        lhs = And(loop[t] == loops[t] for t in hltl_spec.vars)
        rhs = Not(smv_temporal_expr_to_z3(K, 0, hltl_spec.expr, state, loops))
        exprs.append(Implies(lhs, rhs))
    expr = And(exprs)

    for i in range(N-1, first_exists-1, -1):
        if hltl_spec.quantifiers[i] == 'Forall':
            t = hltl_spec.vars[i]
            form = And(behavior(modules[i], state[t], loop[t], init, trans, invar), expr)
            expr = Exists([loop[t]] + [l for j in range(K) for l in state[t][j].values()], form)
        else:
            t = hltl_spec.vars[i]
            form = Implies(behavior(modules[i], state[t], loop[t], init, trans, invar), expr)
            expr = ForAll([loop[t]] + [l for j in range(K) for l in state[t][j].values()], form)
    s.add(expr)
    r = s.check()
    if r == sat:
        print(r)
        print("Counterexample found:")
        model = s.model()
        for i in range(first_exists):
            print(f"* Trace {hltl_spec.vars[i]} *")
            print_trace(model,state[hltl_spec.vars[i]],loop[hltl_spec.vars[i]],K)
    else:
        print(r)
        print("No counterexample found.")