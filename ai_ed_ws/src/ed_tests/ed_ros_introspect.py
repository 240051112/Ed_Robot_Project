#!/usr/bin/env python3
import argparse, csv, json, os, re, shutil, subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- tiny helpers ----------
def esc(s: str) -> str:
    # escape for Graphviz label/ids
    return s.replace("\\", "\\\\").replace('"', '\\"')

def run(cmd: List[str], timeout: float) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def write_text(p: Path, s: str): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(s, encoding="utf-8")
def which(x: str) -> bool: return shutil.which(x) is not None

# ---------- ros2 wrappers ----------
def list_nodes(t: float)->List[str]:
    rc,out,_=run(["ros2","node","list"],t); return [ln.strip() for ln in out.splitlines() if ln.strip()] if rc==0 else []
def list_topics(t: float)->List[str]:
    rc,out,_=run(["ros2","topic","list"],t); return [ln.strip() for ln in out.splitlines() if ln.strip()] if rc==0 else []
def topic_type(topic:str,t: float)->str:
    rc,out,_=run(["ros2","topic","type",topic],t); return out.splitlines()[0].strip() if rc==0 and out else ""
def topic_info_verbose(topic:str,t: float)->Dict[str,List[str]]:
    rc,out,_=run(["ros2","topic","info","--verbose",topic],t); pubs,subs=[],[]
    if rc!=0 or not out: return {"publishers":pubs,"subscribers":subs}
    section=None
    for ln in out.splitlines():
        s=ln.strip()
        ls=s.lower()
        if ls.startswith("publishers"): section="pubs"; continue
        if ls.startswith("subscribers"): section="subs"; continue
        m=re.search(r"Node name:\s*([/\w\-]+)",s)
        if m and section:
            (pubs if section=="pubs" else subs).append(m.group(1))
        elif s.startswith("/") and section:
            (pubs if section=="pubs" else subs).append(s.split()[0])
    return {"publishers":pubs,"subscribers":subs}

def list_services(t: float)->List[Tuple[str,str]]:
    rc,out,_=run(["ros2","service","list","-t"],t)
    if rc!=0: return []
    outp=[]
    for ln in out.splitlines():
        ln=ln.strip()
        if not ln: continue
        m=re.match(r"(\S+)\s+\[(.+)\]",ln)
        outp.append((m.group(1),m.group(2)) if m else (ln,""))
    return outp

def service_info_clients_providers(srv:str,t: float)->Dict[str,List[str]]:
    rc,out,_=run(["ros2","service","info",srv],t); providers,clients=[],[]
    if rc!=0 or not out: return {"providers":providers,"clients":clients}
    current=None
    for ln in out.splitlines():
        s=ln.strip().lower()
        if s.startswith("servers"): current="providers"; continue
        if s.startswith("clients"): current="clients"; continue
        m=re.search(r"Node:\s*([/\w\-]+)",ln)
        if m and current: (providers if current=="providers" else clients).append(m.group(1))
    return {"providers":providers,"clients":clients}

def list_actions(t: float)->List[Tuple[str,str]]:
    rc,out,_=run(["ros2","action","list","-t"],t)
    if rc!=0: return []
    outp=[]
    for ln in out.splitlines():
        ln=ln.strip()
        if not ln: continue
        m=re.match(r"(\S+)\s+\[(.+)\]",ln)
        outp.append((m.group(1),m.group(2)) if m else (ln,""))
    return outp

def action_info_servers_clients(name:str,t: float)->Dict[str,List[str]]:
    rc,out,_=run(["ros2","action","info","-t",name],t); servers,clients=[],[]
    if rc!=0 or not out: return {"servers":servers,"clients":clients}
    section=None
    for ln in out.splitlines():
        s=ln.strip().lower()
        if s.startswith("action servers"): section="servers"; continue
        if s.startswith("action clients"): section="clients"; continue
        m=re.search(r"Node:\s*([/\w\-]+)",ln)
        if m and section: (servers if section=="servers" else clients).append(m.group(1))
    return {"servers":servers,"clients":clients}

def dump_params(node:str,out_dir:Path,t: float):
    rc,out,_=run(["ros2","param","dump",node],t)
    if rc==0 and out: write_text(out_dir/f"{node.strip('/').replace('/', '_')}.yaml",out)

# ---------- graphviz writers ----------
def legend_block(title:str, rows:List[Tuple[str,str]])->str:
    lines=['  legend [shape=none, margin=0, label=<',
           '  <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">',
           f'    <TR><TD COLSPAN="2"><B>{esc(title)}</B></TD></TR>']
    for lab,color in rows:
        lines.append(f'    <TR><TD>{esc(lab)}</TD><TD BGCOLOR="{color}"></TD></TR>')
    lines.append('  </TABLE>','  >];')
    return "\n".join(lines)

def write_dot_nodes_topics(infos:List[Dict], out_dot:Path):
    lines=['digraph ROS2_NodesTopics {','  rankdir=LR;','  node [fontname="Helvetica"];','  edge [fontsize=10];']
    node_set=set(); topic_nodes=set(); edges=[]
    for t in infos:
        topic=t["topic"]; tnode=f'topic_{abs(hash(topic))}'
        topic_nodes.add((tnode,topic,t.get("type","")))
        for n in t["publishers"]: node_set.add(n); edges.append((n,tnode,"pub"))
        for n in t["subscribers"]: node_set.add(n); edges.append((tnode,n,"sub"))
    for n in sorted(node_set):
        lines.append(f'  "{esc(n)}" [shape=box, style=rounded];')
    for tnode,label,ttype in sorted(topic_nodes):
        lab=esc(label); 
        if ttype: lab=f"{lab}\\n({esc(ttype)})"
        lines.append(f'  {tnode} [shape=ellipse, style=filled, fillcolor="#f0f8ff", label="{lab}"];')
    for a,b,kind in edges:
        if kind=="pub":
            lines.append(f'  "{esc(a)}" -> {b} [color="#4e79a7", label="pub"];')
        else:
            lines.append(f'  {a} -> "{esc(b)}" [color="#f28e2b", label="sub"];')
    lines.append(legend_block("Legend",[("Publish (node→topic)","#4e79a7"),("Subscribe (topic→node)","#f28e2b")]))
    lines.append('}')
    write_text(out_dot,"\n".join(lines))

def write_dot_services(infos:List[Dict], out_dot:Path):
    lines=['digraph ROS2_Services {','  rankdir=LR;','  node [fontname="Helvetica"];','  edge [fontsize=10];']
    node_set=set(); svc_nodes=set(); edges=[]
    for s in infos:
        svc=s["service"]; stype=s.get("type",""); snode=f'svc_{abs(hash(svc))}'
        svc_nodes.add((snode,svc,stype))
        for n in s["providers"]: node_set.add(n); edges.append((n,snode,"server"))
        for n in s["clients"]: node_set.add(n); edges.append((snode,n,"client"))
    for n in sorted(node_set):
        lines.append(f'  "{esc(n)}" [shape=box, style=rounded];')
    for snode,label,stype in sorted(svc_nodes):
        lab=esc(label); lab=f"{lab}\\n({esc(stype)})" if stype else lab
        lines.append(f'  {snode} [shape=ellipse, style=filled, fillcolor="#e6f2ff", label="{lab}"];')
    for a,b,kind in edges:
        if kind=="server":
            lines.append(f'  "{esc(a)}" -> {b} [color="#59a14f", label="server"];')
        else:
            lines.append(f'  {a} -> "{esc(b)}" [color="#e15759", label="client"];')
    lines.append(legend_block("Legend",[("Server (node→service)","#59a14f"),("Client (service→node)","#e15759")]))
    lines.append('}')
    write_text(out_dot,"\n".join(lines))

def write_dot_actions(infos:List[Dict], out_dot:Path):
    lines=['digraph ROS2_Actions {','  rankdir=LR;','  node [fontname="Helvetica"];','  edge [fontsize=10];']
    node_set=set(); act_nodes=set(); edges=[]
    for a in infos:
        name=a["action"]; atype=a.get("type",""); anode=f'act_{abs(hash(name))}'
        act_nodes.add((anode,name,atype))
        for n in a["servers"]: node_set.add(n); edges.append((n,anode,"server"))
        for n in a["clients"]: node_set.add(n); edges.append((anode,n,"client"))
    for n in sorted(node_set):
        lines.append(f'  "{esc(n)}" [shape=box, style=rounded];')
    for anode,label,atype in sorted(act_nodes):
        lab=esc(label); lab=f"{lab}\\n({esc(atype)})" if atype else lab
        lines.append(f'  {anode} [shape=ellipse, style=filled, fillcolor="#fff3e6", label="{lab}"];')
    for a,b,kind in edges:
        if kind=="server":
            lines.append(f'  "{esc(a)}" -> {b} [color="#76b7b2", label="server"];')
        else:
            lines.append(f'  {a} -> "{esc(b)}" [color="#edc948", label="client"];')
    lines.append(legend_block("Legend",[("Server (node→action)","#76b7b2"),("Client (action→node)","#edc948")]))
    lines.append('}')
    write_text(out_dot,"\n".join(lines))

def dot_to_png(dot_file:Path, out_png:Path)->bool:
    if not which("dot"): return False
    rc,_,_=run(["dot","-Tpng",str(dot_file),"-o",str(out_png)],timeout=20.0)
    return rc==0

def export_tf_frames(out_dir:Path)->bool:
    if not which("ros2"): return False
    rc,_,_=run(["ros2","run","tf2_tools","view_frames"],timeout=10.0)
    moved=False
    for name in ("frames.pdf","frames.gv"):
        p=Path(name)
        if p.exists(): shutil.move(str(p), str(out_dir / ("tf_"+name))); moved=True
    return moved

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--timeout", type=float, default=1.5, help="Per ros2 call timeout (s)")
    ap.add_argument("--no-params", action="store_true", help="Skip parameter dumps")
    args=ap.parse_args()

    out_root=Path(os.path.expanduser(args.out)).resolve()
    inv_dir=out_root/"inventory"; g_dir=out_root/"graphs"; p_dir=inv_dir/"params"
    ensure_dir(inv_dir); ensure_dir(g_dir); ensure_dir(p_dir)

    # Nodes
    nodes=list_nodes(args.timeout); write_text(inv_dir/"nodes.txt","\n".join(nodes))

    # Topics
    topics=list_topics(args.timeout); topics_info=[]
    with (inv_dir/"topics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["topic","type","publishers","subscribers"])
        for t in topics:
            ttype=topic_type(t,args.timeout); ti=topic_info_verbose(t,args.timeout)
            pubs=sorted(set(ti["publishers"])); subs=sorted(set(ti["subscribers"]))
            topics_info.append({"topic":t,"type":ttype,"publishers":pubs,"subscribers":subs})
            w.writerow([t,ttype,";".join(pubs),";".join(subs)])

    # Services
    services=list_services(args.timeout); services_info=[]
    with (inv_dir/"services.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["service","type","providers","clients"])
        for sname,stype in services:
            si=service_info_clients_providers(sname,args.timeout)
            providers=sorted(set(si["providers"])); clients=sorted(set(si["clients"]))
            services_info.append({"service":sname,"type":stype,"providers":providers,"clients":clients})
            w.writerow([sname,stype,";".join(providers),";".join(clients)])

    # Actions
    actions=list_actions(args.timeout); actions_info=[]
    with (inv_dir/"actions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["action","type","servers","clients"])
        for aname,atype in actions:
            ai=action_info_servers_clients(aname,args.timeout)
            servers=sorted(set(ai["servers"])); clients=sorted(set(ai["clients"]))
            actions_info.append({"action":aname,"type":atype,"servers":servers,"clients":clients})
            w.writerow([aname,atype,";".join(servers),";".join(clients)])

    # Params (optional)
    if not args.no_params:
        for n in nodes:
            try: dump_params(n,p_dir,args.timeout)
            except Exception: pass

    # Summary
    write_text(inv_dir/"summary.json", json.dumps({
        "node_count":len(nodes),"topic_count":len(topics),"service_count":len(services),
        "action_count":len(actions),"timeout_s":args.timeout,"out_dir":str(out_root)
    }, indent=2))

    # Graphs
    dot_topics=g_dir/"graph_nodes_topics.dot"; write_dot_nodes_topics(topics_info,dot_topics); dot_to_png(dot_topics,g_dir/"graph_nodes_topics.png")
    dot_services=g_dir/"graph_services.dot";   write_dot_services(services_info,dot_services); dot_to_png(dot_services,g_dir/"graph_services.png")
    dot_actions=g_dir/"graph_actions.dot";     write_dot_actions(actions_info,dot_actions);   dot_to_png(dot_actions,g_dir/"graph_actions.png")

    tf_ok=export_tf_frames(g_dir)

    print("\n=== ED ROS Introspection ===")
    print(f"Inventory: {inv_dir}")
    print(f"Graphs:    {g_dir}")
    print(f"Params:    {'skipped' if args.no_params else 'dumped (best-effort)'}")
    print(f"TF tree:   {'graphs/tf_frames.pdf' if tf_ok else 'skipped (install ros-humble-tf2-tools)'}")

if __name__=="__main__": main()
