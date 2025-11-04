#!/usr/bin/env bash
# ED 25Q benchmark — single-folder, latency + power + GPU/CPU load + temps
# Uses python3 to parse tegrastats (robust on Jetson mawk/busybox awk).
# Colorful console, live summary, single CSV table.

set -euo pipefail
DEBUG="${DEBUG:-0}" && [[ "$DEBUG" == "1" ]] && set -x

# -------- Config --------
BASE_URL="${1:-http://localhost:8000}"                           # ./test_ed_25.sh <url> <out_dir> <questions>
OUT_DIR="${2:-$(cd "$(dirname "$0")" && pwd)/runs/single}"       # one folder
QUEST_FILE="${3:-$(cd "$(dirname "$0")" && pwd)/questions_25.txt}"

QPS_DELAY="${QPS_DELAY:-0.35}"     # delay between questions (s)
TIMEOUT="${TIMEOUT:-30}"           # /query timeout (s)

TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS:-100}"   # tegrastats cadence
WARMUP_MS="${WARMUP_MS:-400}"                   # wait after start
COOLDOWN_MS="${COOLDOWN_MS:-800}"               # wait before stop

KEEP_RESPONSES="${KEEP_RESPONSES:-1}"           # keep per-Q JSON
KEEP_ENERGY_LOG="${KEEP_ENERGY_LOG:-1}"         # keep tegrastats logs

# -------- Layout --------
mkdir -p "$OUT_DIR"
RESP_DIR="$OUT_DIR/responses"
ENERGY_DIR="$OUT_DIR/energy"
[[ "$KEEP_RESPONSES" == "1" ]] && mkdir -p "$RESP_DIR"
[[ "$KEEP_ENERGY_LOG" == "1" ]] && mkdir -p "$ENERGY_DIR"

CSV_ALL="$OUT_DIR/results.csv"
NDJSON="$OUT_DIR/responses.ndjson"
INDEX_FILE="$OUT_DIR/index.csv"
SUMMARY_FILE="$OUT_DIR/summary.csv"
REPORT_MD="$OUT_DIR/results.md"

# -------- Colors --------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  C_RESET="$(tput sgr0)"; C_DIM="$(tput dim)"; C_BOLD="$(tput bold)"
  C_GR="$(tput setaf 2)"; C_YE="$(tput setaf 3)"; C_BL="$(tput setaf 4)"
  C_MA="$(tput setaf 5)"; C_RE="$(tput setaf 1)"
else
  C_RESET=""; C_DIM=""; C_BOLD=""; C_GR=""; C_YE=""; C_BL=""; C_MA=""; C_RE=""
fi

echo -e "${C_BOLD}Out dir${C_RESET} : $OUT_DIR"
echo -e "${C_BOLD}Server ${C_RESET} : $BASE_URL"

# -------- Sanity --------
for b in curl jq python3; do command -v "$b" >/dev/null 2>&1 || { echo "missing: $b"; exit 1; }; done
command -v tegrastats >/dev/null 2>&1 || { echo -e "${C_RE}ERROR:${C_RESET} tegrastats not found"; exit 1; }

# Probe server
probe_ok=0
curl -s --max-time 5 "$BASE_URL/health" | jq -e '.ok==true' >/dev/null 2>&1 && probe_ok=1 || true
[[ $probe_ok -eq 0 ]] && curl -s --max-time 5 "$BASE_URL/openapi.json" | jq -e '.info.title' >/dev/null 2>&1 && probe_ok=1 || true
[[ $probe_ok -eq 0 ]] && curl -sS --max-time 5 "$BASE_URL/query" -H 'Content-Type: application/json' -d '{"question":"ping"}' | jq -e '.answer' >/dev/null 2>&1 && probe_ok=1 || true
[[ $probe_ok -ne 1 ]] && { echo -e "${C_RE}ERROR:${C_RESET} $BASE_URL not reachable"; exit 1; }

# Questions
[[ -s "$QUEST_FILE" ]] || { echo -e "${C_RE}ERROR:${C_RESET} missing questions: $QUEST_FILE"; exit 1; }
Q_TMP="$(mktemp)"
sed -e 's/\r$//' "$QUEST_FILE" | sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' > "$Q_TMP"
mapfile -t QUESTIONS < "$Q_TMP"; rm -f "$Q_TMP"
QCOUNT="${#QUESTIONS[@]}"; [[ "$QCOUNT" -eq 0 ]] && { echo -e "${C_RE}ERROR:${C_RESET} no questions loaded"; exit 1; }
echo -e "${C_BOLD}Questions${C_RESET}: $QCOUNT"

TIME_BIN=""; [[ -x /usr/bin/time ]] && TIME_BIN="/usr/bin/time" || true

# -------- Helpers --------
msleep(){ local ms="${1:-0}"; [[ "$ms" =~ ^[0-9]+$ ]] || ms=0; sleep "$(awk -v ms="$ms" 'BEGIN{printf("%.3f", ms/1000)}')" ; }

tegrastats_start(){
  local log="$1"
  : > "$log"
  if command -v stdbuf >/dev/null 2>&1; then
    (stdbuf -oL tegrastats --interval "$TEGRA_INTERVAL_MS" > "$log" 2>/dev/null) & echo $!
  else
    (tegrastats --interval "$TEGRA_INTERVAL_MS" > "$log" 2>/dev/null) & echo $!
  fi
}
tegrastats_stop(){ kill "$1" 2>/dev/null || true; }

ensure_samples(){
  local log="$1"
  if [[ ! -s "$log" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout 1s bash -c "tegrastats --interval $TEGRA_INTERVAL_MS >> '$log' 2>/dev/null" || true
    else
      tegrastats --interval "$TEGRA_INTERVAL_MS" >> "$log" 2>/dev/null & local p=$!
      msleep 500; kill "$p" 2>/dev/null || true
    fi
  fi
}

# -------- tegrastats parser (python3; outputs 16 CSV numbers) --------
# avg_in,pk_in,avg_cg,pk_cg,avg_soc,pk_soc,avg_gpu,pk_gpu,avg_cpu,pk_cpu,t_cpu_avg,t_cpu_pk,t_gpu_avg,t_gpu_pk,t_tj_avg,t_tj_pk
tegrastats_parse_all(){
  python3 - "$1" << 'PY'
import re, sys
p = sys.argv[1]
try:
    data = open(p, 'r', errors='ignore').read().splitlines()
except Exception:
    print("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    sys.exit(0)

re_gpu = re.compile(r'GR3D_FREQ\s+(\d+(?:\.\d+)?)%')
re_cpu_block = re.compile(r'CPU\s*\[([^\]]+)\]')
re_core_pct = re.compile(r'(\d+)%@')
re_temp_cpu = re.compile(r'cpu@(\d+(?:\.\d+)?)C')
re_temp_gpu = re.compile(r'gpu@(\d+(?:\.\d+)?)C')
re_temp_tj  = re.compile(r'tj@(\d+(?:\.\d+)?)C')

# rails (support VDD_IN or POM_5V_IN)
re_vdd_in   = re.compile(r'(?:VDD_IN|POM_5V_IN)[^0-9]*([0-9.]+(?:mW|W)?)(?:\s+([0-9.]+(?:mW|W)?))?')
re_vdd_cg   = re.compile(r'VDD_CPU_GPU_CV[^0-9]*([0-9.]+(?:mW|W)?)(?:\s+([0-9.]+(?:mW|W)?))?')
re_vdd_soc  = re.compile(r'VDD_SOC[^0-9]*([0-9.]+(?:mW|W)?)(?:\s+([0-9.]+(?:mW|W)?))?')

def to_watts(tok:str)->float:
    if tok is None: return -1.0
    s = tok.strip().replace(',', '')
    if s.endswith('mW'):
        try: return float(s[:-2]) / 1000.0
        except: return -1.0
    if s.endswith('W'):
        try: return float(s[:-1])
        except: return -1.0
    if '/' in s:  # "12345/..." or "12.3/..."
        s = s.split('/',1)[0]
    try: return float(s)
    except: return -1.0

# accumulators
vi_s=vi_n=0; vi_p=0.0
vc_s=vc_n=0; vc_p=0.0
vs_s=vs_n=0; vs_p=0.0
g_s=g_n=0; g_p=0.0
c_s=c_n=0; c_p=0.0
tc_s=tc_n=0; tc_p=0.0
tg_s=tg_n=0; tg_p=0.0
tt_s=tt_n=0; tt_p=0.0

for line in data:
    m = re_gpu.search(line)
    if m:
        v = float(m.group(1)); g_s += v; g_n += 1; g_p = max(g_p, v)

    m = re_cpu_block.search(line)
    if m:
        blk = m.group(1)
        vals = [int(x) for x in re_core_pct.findall(blk)]
        if vals:
            v = sum(vals)/len(vals)
            c_s += v; c_n += 1; c_p = max(c_p, v)
    else:
        # permissive fallback: scan tokens for N x "%@" patterns
        vals = [int(x) for x in re_core_pct.findall(line)]
        if vals:
            v = sum(vals)/len(vals)
            c_s += v; c_n += 1; c_p = max(c_p, v)

    for r, S, N, P in ((re_temp_cpu, 'tc_s','tc_n','tc_p'),
                       (re_temp_gpu, 'tg_s','tg_n','tg_p'),
                       (re_temp_tj , 'tt_s','tt_n','tt_p')):
        mm = r.search(line)
        if mm:
            v = float(mm.group(1))
            locals()[S] = locals()[S] + v
            locals()[N] = locals()[N] + 1
            locals()[P] = max(locals()[P], v)

    for rex, S, N, P in ((re_vdd_in, 'vi_s','vi_n','vi_p'),
                         (re_vdd_cg, 'vc_s','vc_n','vc_p'),
                         (re_vdd_soc,'vs_s','vs_n','vs_p')):
        mm = rex.search(line)
        if mm:
            v1 = to_watts(mm.group(1))
            if v1 >= 0:
                locals()[S] = locals()[S] + v1
                locals()[N] = locals()[N] + 1
                locals()[P] = max(locals()[P], v1)
            v2 = to_watts(mm.group(2))
            if v2 >= 0:
                locals()[S] = locals()[S] + v2
                locals()[N] = locals()[N] + 1
                locals()[P] = max(locals()[P], v2)

def avg(s, n): return (s / n) if n else 0.0

avi, pvi = avg(vi_s,vi_n), vi_p
avc, pvc = avg(vc_s,vc_n), vc_p
avs, pvs = avg(vs_s,vs_n), vs_p
ag,  pg  = avg(g_s,g_n),  g_p
ac,  pc  = avg(c_s,c_n),  c_p
tca,tcp  = avg(tc_s,tc_n), tc_p
tga,tgp  = avg(tg_s,tg_n), tg_p
tja,tjp  = avg(tt_s,tt_n), tt_p

print(f"{avi:.3f},{pvi:.3f},{avc:.3f},{pvc:.3f},{avs:.3f},{pvs:.3f},"
      f"{ag:.2f},{pg:.2f},{ac:.2f},{pc:.2f},"
      f"{tca:.2f},{tcp:.2f},{tga:.2f},{tgp:.2f},{tja:.2f},{tjp:.2f}")
PY
}

lint_security(){
  local json="$1"
  local ans; ans="$(jq -r '.answer // ""' "$json" 2>/dev/null || echo "")"
  local urls pii
  urls=$({ grep -Eo 'https?://[^ ]+' <<<"$ans" || true; } | wc -l | tr -d ' ')
  pii=$({ grep -Eo '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b' <<<"$ans" || true; } | wc -l | tr -d ' ')
  echo "${urls:-0},${pii:-0}"
}

run_timed_curl(){
  local payload="$1" out="$2" tfile="$3"
  if [[ -n "$TIME_BIN" ]]; then
    "$TIME_BIN" -f 'elapsed_s=%E user_s=%U sys_s=%S' \
      curl -sS --max-time "$TIMEOUT" "$BASE_URL/query" \
        -H 'Content-Type: application/json' -d "$payload" -o "$out" 2>"$tfile" || true
  else
    { time curl -sS --max-time "$TIMEOUT" "$BASE_URL/query" \
        -H 'Content-Type: application/json' -d "$payload" -o "$out"; } 2>"$tfile" || true
  fi
}

# numeric-coercion for printf (turn empty/NA into 0)
num0(){ local v="${1:-0}"; [[ "$v" =~ ^[0-9.]+$ ]] || v=0; printf "%s" "$v"; }

# -------- CSVs --------
echo "qid,question,mode,topic,ctx_count,total_s,answer_len,sources_count,avg_vdd_in_w,peak_vdd_in_w,avg_cpu_gpu_cv_w,peak_cpu_gpu_cv_w,avg_vdd_soc_w,peak_vdd_soc_w,avg_gpu_pct,peak_gpu_pct,avg_cpu_pct,peak_cpu_pct,avg_temp_cpu_c,peak_temp_cpu_c,avg_temp_gpu_c,peak_temp_gpu_c,avg_temp_tj_c,peak_temp_tj_c,url_count,pii_count" > "$CSV_ALL"
: > "$NDJSON"
echo "qid,question" > "$INDEX_FILE"
printf "metric,value\n%s\n%s\n%s\n%s\n" "total_requests,0" "avg_latency_s,NA" "avg_power_vdd_in_w,NA" "avg_peak_power_vdd_in_w,NA" > "$SUMMARY_FILE"

refresh_summary(){
  local total avg_latency avg_pwr avg_peak
  total=$(awk 'END{print NR-1}' "$CSV_ALL")
  avg_latency=$(awk -F, 'NR>1{ s+=$6; n++ } END{ if(n) printf "%.3f", s/n; else print "NA" }' "$CSV_ALL")
  avg_pwr=$(awk -F, 'NR>1{ if($9  != "NA" && $9  != "") {s+=$9;  n++} } END{ if(n) printf "%.3f", s/n; else print "NA" }' "$CSV_ALL")
  avg_peak=$(awk -F, 'NR>1{ if($10 != "NA" && $10 != "") {s+=$10; n++} } END{ if(n) printf "%.3f", s/n; else print "NA" }' "$CSV_ALL")
  {
    echo "metric,value"
    echo "total_requests,$total"
    echo "avg_latency_s,$avg_latency"
    echo "avg_power_vdd_in_w,$avg_pwr"
    echo "avg_peak_power_vdd_in_w,$avg_peak"
  } > "$SUMMARY_FILE"
}

# -------- Main loop --------
for i in "${!QUESTIONS[@]}"; do
  idx=$((i+1)); qid=$(printf 'Q%02d' "$idx"); q="${QUESTIONS[$i]}"
  out_json="${RESP_DIR}/${qid}.json"; [[ "$KEEP_RESPONSES" == "1" ]] || out_json="$(mktemp)"
  tfile="${OUT_DIR}/${qid}.time"
  [[ "$KEEP_ENERGY_LOG" == "1" ]] && log_path="${ENERGY_DIR}/${qid}.tegrastats.log" || log_path="$(mktemp)"

  payload=$(jq -Rn --arg q "$q" '{question:$q}')

  pid="$(tegrastats_start "$log_path")"
  msleep "$WARMUP_MS"
  run_timed_curl "$payload" "$out_json" "$tfile"
  msleep "$COOLDOWN_MS"
  tegrastats_stop "$pid"
  ensure_samples "$log_path"

  # valid JSON
  jq -e type "$out_json" >/dev/null 2>&1 || echo "{\"error\":\"invalid or empty response\"}" > "$out_json"

  mode=$(jq -r '.mode // "?"' "$out_json" 2>/dev/null || echo "?")
  topic=$(jq -r '.topic_area // "?"' "$out_json" 2>/dev/null || echo "?")
  ctx=$(jq -r '.ctx_count // 0' "$out_json" 2>/dev/null || echo "0")
  total_s=$(jq -r '.timing.total_s // 0' "$out_json" 2>/dev/null || echo "0")
  ans_len=$(jq -r '(.answer // "") | length' "$out_json" 2>/dev/null || echo "0")
  src_cnt=$(jq -r '(.sources // []) | length' "$out_json" 2>/dev/null || echo "0")

  IFS=, read -r avg_in pk_in avg_cg pk_cg avg_soc pk_soc avg_gpu pk_gpu avg_cpu pk_cpu tca tcp tga tgp tja tjp <<<"$(tegrastats_parse_all "$log_path")"
  IFS=, read -r url_cnt pii_cnt <<<"$(lint_security "$out_json")"

  echo "${qid},${q//,/;},$mode,$topic,$ctx,$total_s,$ans_len,$src_cnt,$avg_in,$pk_in,$avg_cg,$pk_cg,$avg_soc,$pk_soc,$avg_gpu,$pk_gpu,$avg_cpu,$pk_cpu,$tca,$tcp,$tga,$tgp,$tja,$tjp,$url_cnt,$pii_cnt" >> "$CSV_ALL"
  jq -c '.' "$out_json" >> "$NDJSON" || true
  echo "${qid},${q//,/;}" >> "$INDEX_FILE"

  [[ "$KEEP_RESPONSES" == "1" ]] || rm -f "$out_json"
  [[ "$KEEP_ENERGY_LOG" == "1" ]] || rm -f "$log_path"

  # pretty line
  printf "%s%s%s %s/ ctx=%s / %s%.3fs%s / %s%.2fW%s→%s%.2fW%s / %sGPU %.0f%%%s pk %s%.0f%%%s / %sCPU %.0f%%%s pk %s%.0f%%%s / %sTcpu %.1f°%sC pk %s%.1f°%sC%s\n" \
    "$C_BOLD" "$qid" "$C_RESET" "$mode" "$ctx" \
    "$C_BL" "$(num0 "$total_s")" "$C_RESET" \
    "$C_GR" "$(num0 "$avg_in")" "$C_RESET" "$C_GR" "$(num0 "$pk_in")" "$C_RESET" \
    "$C_MA" "$(num0 "$avg_gpu")" "$C_RESET" "$C_MA" "$(num0 "$pk_gpu")" "$C_RESET" \
    "$C_YE" "$(num0 "$avg_cpu")" "$C_RESET" "$C_YE" "$(num0 "$pk_cpu")" "$C_RESET" \
    "$C_DIM" "$(num0 "$tca")" "$C_RESET" "$C_DIM" "$(num0 "$tcp")" "$C_RESET" "$C_RESET"

  refresh_summary
  sleep "$QPS_DELAY"
done

# -------- Final report --------
refresh_summary
{
  echo "# ED 25Q Benchmark"
  echo
  echo "## Summary"
  sed '1d;s/^/- /' "$SUMMARY_FILE"
  echo
  echo "## Files"
  echo "- \`$CSV_ALL\` (full table)"
  echo "- \`$NDJSON\` (raw responses, ndjson)"
  [[ "$KEEP_RESPONSES" == "1" ]] && echo "- responses/ (per-Q JSONs)"
  [[ "$KEEP_ENERGY_LOG" == "1" ]] && echo "- energy/ (tegrastats logs)"
} > "$REPORT_MD"

echo
echo -e "${C_BOLD}Done.${C_RESET} Outputs in: $OUT_DIR"
echo "  Table : $CSV_ALL"
echo "  Raw   : $NDJSON"
echo "  Summ  : $SUMMARY_FILE"
echo "  Report: $REPORT_MD"
