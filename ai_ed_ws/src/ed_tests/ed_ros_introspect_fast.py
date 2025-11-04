import time

def topic_info_verbose(topic: str, timeout: float) -> Dict[str, List[str]]:
    """
    Return {publishers:[/node], subscribers:[/node]} by trying:
      1) ros2 topic info --verbose <topic>
      2) ros2 topic info -v <topic>
      3) ros2 topic info <topic>  (fallback parser)
    Works across Humble variants that format the output differently.
    """
    pubs, subs = [], []

    def parse_verbose(text: str):
        local_pubs, local_subs = [], []
        section = None
        for ln in text.splitlines():
            s = ln.strip()
            low = s.lower()
            if low.startswith("publishers"):
                section = "pubs";  continue
            if low.startswith("subscribers"):
                section = "subs";  continue
            # Common forms:
            #   Node name: /node_x
            m = re.search(r"Node name:\s*([/\w\-\.]+)", s)
            if m and section:
                (local_pubs if section == "pubs" else local_subs).append(m.group(1))
                continue
            # Some builds print bare node lines (start with '/')
            if s.startswith("/") and section:
                (local_pubs if section == "pubs" else local_subs).append(s.split()[0])
        return local_pubs, local_subs

    # Try --verbose
    rc, out, _ = run(["ros2", "topic", "info", "--verbose", topic], timeout)
    if rc == 0 and out:
        p, s = parse_verbose(out)
        if p or s:
            return {"publishers": sorted(set(p)), "subscribers": sorted(set(s))}

    # Try -v
    rc, out, _ = run(["ros2", "topic", "info", "-v", topic], timeout)
    if rc == 0 and out:
        p, s = parse_verbose(out)
        if p or s:
            return {"publishers": sorted(set(p)), "subscribers": sorted(set(s))}

    # Last resort: plain info (no -v). Humble prints “Publisher count: …” then lines like:
    #   Node name: /node_x
    rc, out, _ = run(["ros2", "topic", "info", topic], timeout)
    if rc == 0 and out:
        p, s = [], []
        cur = None
        for ln in out.splitlines():
            L = ln.strip().lower()
            if L.startswith("publisher"):
                cur = "pubs"; continue
            if L.startswith("subscriber"):
                cur = "subs"; continue
            m = re.search(r"Node name:\s*([/\w\-\.]+)", ln)
            if m and cur:
                (p if cur == "pubs" else s).append(m.group(1))
        if p or s:
            return {"publishers": sorted(set(p)), "subscribers": sorted(set(s))}

    return {"publishers": [], "subscribers": []}


def write_dot_nodes_topics(topics_info: List[Dict], out_dot: Path):
    """
    Draw nodes (boxes) and topics (ellipses) with pub/sub edges and a small legend.
    Ensures nodes actually live inside their cluster subgraphs (so layout is tidy).
    """
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        'digraph ROS2_NodesTopics {',
        '  rankdir=LR;',
        '  node [fontname="Helvetica"];',
        '  edge [fontsize=10];'
    ]

    node_set = set()
    topic_nodes = []   # [(id, label, type)]
    edges = []         # [(src, dst, kind)] where kind in {"pub","sub"}

    for t in topics_info:
        topic = t["topic"]
        ttype = t.get("type", "")
        tnode = f'topic_{abs(hash(topic))}'
        topic_nodes.append((tnode, topic, ttype))
        for n in t.get("publishers", []):
            node_set.add(n);  edges.append((f'"{esc(n)}"', tnode, "pub"))
        for n in t.get("subscribers", []):
            node_set.add(n);  edges.append((tnode, f'"{esc(n)}"', "sub"))

    # clusters with contents
    lines.append('  subgraph cluster_nodes { label="Nodes"; color=lightgrey; style=dashed;')
    for n in sorted(node_set):
        lines.append(f'    "{esc(n)}" [shape=box, style=rounded];')
    lines.append('  }')

    lines.append('  subgraph cluster_topics { label="Topics"; color=lightblue; style=dashed;')
    for tnode, label, ttype in sorted(topic_nodes):
        lab = esc(label)
        if ttype:
            lab = f"{lab}\\n({esc(ttype)})"
        lines.append(f'    {tnode} [shape=ellipse, style=filled, fillcolor="#f0f8ff", label="{lab}"];')
    lines.append('  }')

    # edges (blue for pub, orange for sub)
    for a, b, kind in edges:
        color = "#4e79a7" if kind == "pub" else "#f28e2b"
        lines.append(f'  {a} -> {b} [color="{color}", label="{kind}"];')

    # legend
    lines += [
        '  legend [shape=none, margin=0, label=<',
        '  <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">',
        '    <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>',
        '    <TR><TD>Publish (node→topic)</TD><TD BGCOLOR="#4e79a7"></TD></TR>',
        '    <TR><TD>Subscribe (topic→node)</TD><TD BGCOLOR="#f28e2b"></TD></TR>',
        '  </TABLE>',
        '  >];'
    ]

    lines.append('}')
    write_text(out_dot, "\n".join(lines))
