# ===============================================
# Full Multi-Domain Symbolic Engine (Python)
# ===============================================

from functools import total_ordering

# -----------------------------------------------
# 1. Expression Tree / Nodes
# -----------------------------------------------
@total_ordering
class ExprNode:
    def __init__(self, op, children=None, domain="arithmetic", constraints=None):
        self.op = op
        self.children = children or []
        self.domain = domain
        self.constraints = constraints or set()

    def __repr__(self):
        if not self.children:
            return str(self.op)
        return f"{self.op}({', '.join(map(str, self.children))})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def copy(self):
        return ExprNode(
            self.op,
            [c.copy() for c in self.children],
            domain=self.domain,
            constraints=set(self.constraints)
        )

# -----------------------------------------------
# 2. Pattern Matching + Rewrite Rules
# -----------------------------------------------
class PatternVar:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"?{self.name}"

def match(pattern, node, bindings=None):
    if bindings is None:
        bindings = {}
    if isinstance(pattern, PatternVar):
        if pattern.name in bindings:
            return None if bindings[pattern.name] != node else bindings
        bindings[pattern.name] = node
        return bindings
    if pattern.op != node.op or len(pattern.children) != len(node.children):
        return None
    for p_child, n_child in zip(pattern.children, node.children):
        bindings = match(p_child, n_child, bindings)
        if bindings is None:
            return None
    return bindings

def substitute(template, bindings):
    if isinstance(template, PatternVar):
        return bindings[template.name].copy()
    new_node = ExprNode(template.op, domain=template.domain)
    new_node.constraints = set(template.constraints)
    new_node.children = [substitute(c, bindings) for c in template.children]
    return new_node

class RewriteRule:
    def __init__(self, pattern, replacement, condition=None):
        self.pattern = pattern
        self.replacement = replacement
        self.condition = condition  # function(bindings) -> bool
    def apply(self, node):
        bindings = match(self.pattern, node)
        if bindings is None:
            return None
        if self.condition and not self.condition(bindings):
            return None
        return substitute(self.replacement, bindings)

# -----------------------------------------------
# 3. Canonicalization
# -----------------------------------------------
class Canonicalizer:
    COMMUTATIVE = {"Add", "Mul", "AND", "OR"}
    ASSOCIATIVE = {"Add", "Mul", "AND", "OR"}

    def canonicalize(self, node):
        node = self._canonicalize_children(node)
        # Flatten associative ops
        if node.op in self.ASSOCIATIVE:
            flat_children = []
            for c in node.children:
                if c.op == node.op:
                    flat_children.extend(c.children)
                else:
                    flat_children.append(c)
            node.children = flat_children
        # Sort for commutative ops
        if node.op in self.COMMUTATIVE:
            node.children.sort(key=lambda x: repr(x))
        # Constant folding
        node = self.fold_constants(node)
        # Logic normalization
        node = self.logic_nnf(node)
        return node

    def _canonicalize_children(self, node):
        new_node = node.copy()
        new_node.children = [self.canonicalize(c) for c in node.children]
        return new_node

    def fold_constants(self, node):
        ints = [c for c in node.children if isinstance(c.op, (int,float))]
        non_ints = [c for c in node.children if not isinstance(c.op,(int,float))]
        if not ints:
            return node
        if node.op == "Add":
            return ExprNode(sum(c.op for c in ints), non_ints)
        if node.op == "Mul":
            prod = 1
            for c in ints:
                prod *= c.op
            return ExprNode(prod, non_ints)
        return node

    def logic_nnf(self, node):
        if node.op == "NOT" and node.children:
            child = node.children[0]
            if child.op == "NOT":
                return self.logic_nnf(child.children[0])
            if child.op == "AND":
                return ExprNode("OR", [ExprNode("NOT",[c]) for c in child.children], domain="logic")
            if child.op == "OR":
                return ExprNode("AND",[ExprNode("NOT",[c]) for c in child.children], domain="logic")
        return node

# -----------------------------------------------
# 4. Constraint Propagation
# -----------------------------------------------
def propagate_constraints(node):
    new_node = node.copy()
    for i,c in enumerate(new_node.children):
        new_node.children[i] = propagate_constraints(c)
    # Arithmetic constraints
    if new_node.op == "Div":
        denom = new_node.children[1]
        new_node.constraints.add(f"{repr(denom)} != 0")
    if new_node.op == "Sqrt":
        arg = new_node.children[0]
        new_node.constraints.add(f"{repr(arg)} >= 0")
    if new_node.op == "Pow":
        base, exp = new_node.children
        if isinstance(exp.op,(int,float)) and not float(exp.op).is_integer():
            new_node.constraints.add(f"{repr(base)} >= 0")
    return new_node

# -----------------------------------------------
# 5. Domain Theories
# -----------------------------------------------
class DomainTheory:
    def validate(self,node:ExprNode) -> bool:
        return True

class ArithmeticTheory(DomainTheory):
    def validate(self,node):
        for c in node.constraints:
            if "!= 0" in c and "0" in repr(node):
                return False
            if ">=" in c and isinstance(node.op,(int,float)) and node.op<0:
                return False
        return True

class LogicTheory(DomainTheory):
    def validate(self,node):
        for c in node.constraints:
            if c=="A AND NOT A":
                return False
        return True

class MultiDomainTheory(DomainTheory):
    def __init__(self,*theories):
        self.theories = theories
    def validate(self,node):
        return all(t.validate(node) for t in self.theories)

# -----------------------------------------------
# 6. Rewrite Engine with Domain Checks
# -----------------------------------------------
class RewriteEngine:
    def __init__(self,rules,domain_theory:DomainTheory):
        self.rules = rules
        self.domain_theory = domain_theory

    def rewrite(self,node:ExprNode):
        node = propagate_constraints(node)
        for i,c in enumerate(node.children):
            node.children[i] = self.rewrite(c)
        for rule in self.rules:
            new_node = rule.apply(node)
            if new_node:
                new_node = propagate_constraints(new_node)
                if self.domain_theory.validate(new_node):
                    return self.rewrite(new_node)
        return node

# -----------------------------------------------
# 7. E-Graph
# -----------------------------------------------
class EGraph:
    def __init__(self):
        self.classes = {}
        self.memo = {}
        self.next_id = 0

    def add(self,node):
        if node in self.memo:
            return self.memo[node]
        eid = self.next_id
        self.next_id +=1
        self.classes[eid] = {node}
        self.memo[node] = eid
        return eid

    def merge(self,id1,id2):
        if id1==id2:
            return
        self.classes[id1] |= self.classes[id2]
        del self.classes[id2]
        for n,eid in list(self.memo.items()):
            if eid==id2:
                self.memo[n]=id1

    def rebuild(self):
        new_map={}
        for eid,nodes in self.classes.items():
            for n in nodes:
                new_map[n]=eid
        self.memo = new_map

def equality_saturate(egraph,rules,max_iterations=50):
    for _ in range(max_iterations):
        new_merges=[]
        for eid,nodes in list(egraph.classes.items()):
            for n in list(nodes):
                for rule in rules:
                    bindings = match(rule.pattern,n)
                    if bindings:
                        new_node = substitute(rule.replacement,bindings)
                        new_id = egraph.add(new_node)
                        new_merges.append((eid,new_id))
        if not new_merges:
            break
        for id1,id2 in new_merges:
            egraph.merge(id1,id2)
        egraph.rebuild()

# -----------------------------------------------
# 8. Automatic Differentiation
# -----------------------------------------------
class DiffNode(ExprNode):
    def derivative(self,var):
        if self.op=="Add":
            return ExprNode("Add",[c.derivative(var) for c in self.children])
        if self.op=="Mul":
            terms=[]
            for i,c in enumerate(self.children):
                others=[self.children[j] for j in range(len(self.children)) if j!=i]
                terms.append(ExprNode("Mul",[c.derivative(var)]+others))
            return ExprNode("Add",terms)
        if self.op==var:
            return ExprNode(1)
        return ExprNode(0)
  
