"""Graful computațional al unei valori, scris în limbajul DOT și desenat cu Graphviz."""


def trace(root):
    nodes, edges = [], []
    seen = set()

    def build(v):
        if v in seen:
            return
        seen.add(v)
        nodes.append(v)
        for child in v._prev:
            edges.append((child, v))
            build(child)

    build(root)
    return nodes, edges


def to_dot(root, rankdir="LR"):
    nodes, edges = trace(root)
    ids = {v: f"v{i}" for i, v in enumerate(nodes)}
    lines = [
        "digraph {",
        f"  rankdir={rankdir}; nodesep=0.15; ranksep=0.3;",
        '  node [fontname="Helvetica", fontsize=11];',
    ]
    for v in nodes:
        name = f"{v.label} | " if v.label else ""
        fields = f"{{ data {v.data:.4f} | grad {v.grad:.4f} }}"
        lines.append(f'  {ids[v]} [shape=record, label="{{ {name}{fields} }}"];')
        if v._op:
            lines.append(f'  {ids[v]}_op [label="{v._op}", width=0.4, height=0.3];')
            lines.append(f"  {ids[v]}_op -> {ids[v]};")
    for child, parent in edges:
        lines.append(f"  {ids[child]} -> {ids[parent]}_op;")
    lines.append("}")
    return "\n".join(lines)


def draw_graph(root, path=None, rankdir="LR"):
    import graphviz

    graph = graphviz.Source(to_dot(root, rankdir))
    if path:
        graph.render(outfile=path, cleanup=True)
    return graph
