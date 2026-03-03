import random
import math
from collections import defaultdict

# ------------------------------
# Expression Tree / Nodes
# ------------------------------
class ExprNode:
    """
    Multi-domain symbolic expression node.
    domain: "arithmetic" or "logic"
    """
    def __init__(self, op, children=None, domain="arithmetic"):
        self.op = op
        self.children = children or []
        self.domain = domain

    def __repr__(self):
        if not self.children:
            return str(self.op)
        return f"{self.op}({', '.join(map(str, self.children))})"

# For pattern variables in rewrites
class PatternVar:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"?{self.name}"

# Helper constructors for symbols
def Symbol(name, domain="arithmetic"):
    return ExprNode(name, domain=domain)

# ------------------------------
# Pattern-Matching and Substitution
# ------------------------------
def match(pattern, node, bindings=None):
    if bindings is None:
        bindings = {}
    if isinstance(pattern, PatternVar):
        if pattern.name in bindings:
            if bindings[pattern.name] == node:
                return bindings
            return None
        bindings[pattern.name] = node
        return bindings
    if not pattern.children:
        return bindings if pattern.op == node.op else None
    if pattern.op != node.op or len(pattern.children) != len(node.children):
        return None
    for p_child, n_child in zip(pattern.children, node.children):
        bindings = match(p_child, n_child, bindings)
        if bindings is None:
            return None
    return bindings

def substitute(template, bindings):
    if isinstance(template, PatternVar):
        return bindings[template.name]
    if not template.children:
        return ExprNode(template.op, domain=template.domain)
    return ExprNode(
        template.op,
        [substitute(child, bindings) for child in template.children],
        domain=template.domain
    )

class RewriteRule:
    def __init__(self, pattern, replacement, name=None, condition=None):
        self.pattern = pattern
        self.replacement = replacement
        self.name = name or "UnnamedRule"
        self.condition = condition
    def apply(self, node):
        bindings = match(self.pattern, node)
        if bindings is not None and (self.condition is None or self.condition(bindings)):
            return substitute(self.replacement, bindings)
        return None

class RewriteEngine:
    def __init__(self, rules):
        self.rules = rules
    def rewrite(self, node):
        if node.children:
            node = ExprNode(
                node.op,
                [self.rewrite(child) for child in node.children],
                domain=node.domain
            )
        for rule in self.rules:
            new_node = rule.apply(node)
            if new_node:
                return self.rewrite(new_node)
        return node

# ------------------------------
# Canonicalization
# ------------------------------
class Canonicalizer:
    COMMUTATIVE_OPS = {"Add", "Multiply", "AND", "OR"}
    ASSOCIATIVE_OPS = {"Add", "Multiply", "AND", "OR"}

    def canonicalize(self, node):
        if not node.children:
            return node
        node.children = [self.canonicalize(c) for c in node.children]
        # Flatten associative ops
        if node.op in self.ASSOCIATIVE_OPS:
            flattened = []
            for child in node.children:
                if child.op == node.op:
                    flattened.extend(child.children)
                else:
                    flattened.append(child)
            node.children = flattened
        # Sort commutative children
        if node.op in self.COMMUTATIVE_OPS:
            node.children = sorted(node.children, key=lambda x: repr(x))
        # Constant folding (arithmetic only)
        if node.domain == "arithmetic":
            if node.op == "Add":
                return self.fold_add(node)
            if node.op == "Multiply":
                return self.fold_multiply(node)
        # Logical simplification: basic NNF
        if node.domain == "logic":
            node = self.logic_nnf(node)
        return node

    def fold_add(self, node):
        total = 0
        new_children = []
        for c in node.children:
            if isinstance(c.op, (int, float)):
                total += c.op
            else:
                new_children.append(c)
        if total != 0:
            new_children.append(ExprNode(total))
        if not new_children:
            return ExprNode(0)
        if len(new_children) == 1:
            return new_children[0]
        return ExprNode("Add", new_children)

    def fold_multiply(self, node):
        total = 1
        new_children = []
        for c in node.children:
            if isinstance(c.op, (int, float)):
                total *= c.op
            else:
                new_children.append(c)
        if total == 0:
            return ExprNode(0)
        if total != 1:
            new_children.append(ExprNode(total))
        if not new_children:
            return ExprNode(1)
        if len(new_children) == 1:
            return new_children[0]
        return ExprNode("Multiply", new_children)

    def logic_nnf(self, node):
        # Push NOTs down (De Morgan)
        if node.op == "NOT" and node.children[0].op == "NOT":
            return self.logic_nnf(node.children[0].children[0])
        if node.op == "NOT" and node.children[0].op == "AND":
            return ExprNode("OR", [self.logic_nnf(ExprNode("NOT", [c], "logic")) for c in node.children[0].children], "logic")
        if node.op == "NOT" and node.children[0].op == "OR":
            return ExprNode("AND", [self.logic_nnf(ExprNode("NOT", [c], "logic")) for c in node.children[0].children], "logic")
        return node

# ------------------------------
# Commutativity Discovery
# ------------------------------
class CommutativityAnalyzer:
    def __init__(self, trials=20):
        self.trials = trials
        self.cache = {}
    def discover(self, A, B):
        key = tuple(sorted([A.op, B.op]))
        if key in self.cache:
            return self.cache[key]
        if getattr(A, "side_effects", False) or getattr(B, "side_effects", False):
            self.cache[key] = False
            return False
        # Random property testing
        for _ in range(self.trials):
            x = random.uniform(-100, 100)
            try:
                ab = A.forward(B.forward(x))
                ba = B.forward(A.forward(x))
            except Exception:
                self.cache[key] = False
                return False
            if abs(ab - ba) > 1e-9:
                self.cache[key] = False
                return False
        self.cache[key] = True
        return True

# ------------------------------
# Constraint-Safe Reorder Engine
# ------------------------------
class OptimizableStage:
    def __init__(self, name, forward, cost=1, dependencies=None, side_effects=False):
        self.name = name
        self.forward = forward
        self.cost = cost
        self.dependencies = dependencies or set()
        self.side_effects = side_effects
        self.commutes_with = set()
    def run(self, x):
        return self.forward(x)

class ConstraintSolver:
    def build_graph(self, stages):
        graph = {s.name: set() for s in stages}
        name_to_stage = {s.name: s for s in stages}
        for s in stages:
            for dep in s.dependencies:
                graph[dep].add(s.name)
        for i, A in enumerate(stages):
            for j, B in enumerate(stages):
                if i >= j:
                    continue
                if not self.can_commute(A, B):
                    graph[A.name].add(B.name)
        for i in range(len(stages)-1):
            A, B = stages[i], stages[i+1]
            if A.side_effects or B.side_effects:
                graph[A.name].add(B.name)
        return graph

    def can_commute(self, A, B):
        return CommutativityAnalyzer().discover(A, B)

    def solve(self, stages):
        graph = self.build_graph(stages)
        in_degree = {k:0 for k in graph}
        for deps in graph.values():
            for node in deps:
                in_degree[node] +=1
        ready = sorted([s for s in stages if in_degree[s.name]==0], key=lambda s:s.cost)
        result = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for neighbor in graph[current.name]:
                in_degree[neighbor]-=1
                if in_degree[neighbor]==0:
                    stage_obj = next(s for s in stages if s.name==neighbor)
                    ready.append(stage_obj)
                    ready = sorted(ready,key=lambda s:s.cost)
        if len(result)!=len(stages):
            raise ValueError("Cycle detected")
        return result

# ------------------------------
# E-Graph / Equality Saturation
# ------------------------------
class ENode:
    def __init__(self, op, children):
        self.op = op
        self.children = tuple(children)
    def __hash__(self):
        return hash((self.op, self.children))
    def __eq__(self, other):
        return self.op==other.op and self.children==other.children

class EClass:
    def __init__(self):
        self.nodes = set()
        self.parents = set()

class EGraph:
    def __init__(self):
        self.classes = {}
        self.memo = {}
        self.next_id = 0
    def add(self, node):
        if node in self.memo:
            return self.memo[node]
        eid = self.next_id
        self.next_id += 1
        self.memo[node] = eid
        self.classes[eid] = EClass()
        self.classes[eid].nodes.add(node)
        return eid
    def merge(self, id1, id2):
        if id1==id2: return
        self.classes[id1].nodes |= self.classes[id2].nodes
        del self.classes[id2]
        for node, eid in list(self.memo.items()):
            if eid==id2:
                self.memo[node]=id1

def enode_to_tree(enode):
    return ExprNode(enode.op, list(enode.children))

def tree_to_enode(tree):
    return ENode(tree.op, tree.children)

def equality_saturate(egraph, rules, iterations=10):
    for _ in range(iterations):
        new_equalities = []
        for eid, eclass in list(egraph.classes.items()):
            for node in list(eclass.nodes):
                expr_tree = enode_to_tree(node)
                for rule in rules:
                    bindings = match(rule.pattern, expr_tree)
                    if bindings:
                        new_expr = substitute(rule.replacement, bindings)
                        new_node = tree_to_enode(new_expr)
                        new_id = egraph.add(new_node)
                        new_equalities.append((eid, new_id))
        for id1,id2 in new_equalities:
            egraph.merge(id1,id2)

def extract_best(egraph):
    best = {}
    def cost(node):
        if not node.children: return 1
        return 1 + sum(cost(c) for c in node.children)
    for eid, eclass in egraph.classes.items():
        best_node = min(eclass.nodes, key=lambda n: cost(enode_to_tree(n)))
        best[eid] = best_node
    return best

# ------------------------------
# Automatic Differentiation
# ------------------------------
class DiffNode(ExprNode):
    def derivative(self, var):
        if self.op == "Add":
            return DiffNode("Add",[c.derivative(var) for c in self.children])
        if self.op == "Subtract":
            return DiffNode("Subtract",[c.derivative(var) for c in self.children])
        if self.op == "Multiply":
            terms = []
            for i,c in enumerate(self.children):
                others = [self.children[j] for j in range(len(self.children)) if j!=i]
                terms.append(DiffNode("Multiply",[c.derivative(var)]+others))
            return DiffNode("Add",terms)
        if self.op == "Divide":
            u,v = self.children
            return DiffNode("Divide",[DiffNode("Subtract",[DiffNode("Multiply",[u.derivative(var),v]),DiffNode("Multiply",[u,v.derivative(var)])]),DiffNode("Multiply",[v,v])])
        if self.op == "Pow":
            base, exponent = self.children
            return DiffNode("Multiply",[DiffNode("Pow",[base,exponent]),DiffNode("Add",[DiffNode("Multiply",[exponent.derivative(var),DiffNode("Log",[base])]),DiffNode("Multiply",[exponent,DiffNode("Divide",[base.derivative(var),base])])])])
        if self.op == var:
            return DiffNode(1)
        if not self.children:
            return DiffNode(0)
        return DiffNode(0)
                 
