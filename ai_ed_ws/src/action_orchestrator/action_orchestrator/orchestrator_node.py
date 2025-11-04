#!/usr/bin/env python3
import os, re, time, json, requests
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from dofbot_pro_interface.msg import SpeechCommand
from dofbot_pro_interface.action import SkillExecution
from dofbot_pro_interface.srv import Speak

# ---------------- Config ----------------
RAG_URL = os.getenv("ED_RAG_URL", "http://localhost:8000/query")
HTTP_TIMEOUT = float(os.getenv("ED_RAG_TIMEOUT", "15.0"))
TIMING_ENABLED = os.getenv("ED_TIMING", "0") == "1"   # <-- turn on with: export ED_TIMING=1

# classify “question” by leading WH words or trailing '?'
WH_WORDS = ("what","who","where","when","why","how","is","are","do","does","can","should","tell","define","explain")

def strip_markdown(text: str) -> str:
    text = re.sub(r'[\*_`]', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if t.endswith("?") or any(t.startswith(w + " ") for w in WH_WORDS):
        return "question"
    return "command"

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("ed_orchestrator_node")

        # subs/clients
        self.subscription = self.create_subscription(
            SpeechCommand, "/ed/speech_command", self.command_callback, 10
        )
        self._action_client = ActionClient(self, SkillExecution, "ed_skill_server")
        self.tts_client = self.create_client(Speak, "/ed/speak")

        # allow a tiny skill vocab
        self.allowed_skills = {"go_home", "open_gripper", "close_gripper"}

        # optional profanity guard
        self._bad = re.compile(r"\b(fuck|dick|bitch)\b", re.IGNORECASE)

        self.get_logger().info("Ed's Orchestrator is running. Waiting for voice command...")

    # ------------- Core routing -------------
    def command_callback(self, msg: SpeechCommand):
        t0 = time.perf_counter()
        text = (msg.command or "").strip()
        if not text:
            return

        if self._bad.search(text):
            self.speak("Let’s keep it professional. What task do you need on the line?")
            return

        self.get_logger().info(f"Orchestrator received text: '{text}'")

        try:
            intent = classify_intent(text)
            self.get_logger().info(f"Intent: '{intent}'")

            if intent == "command":
                self._do_skill(text.lower(), t0)
            else:
                self._do_question(text, t0)

        except Exception as e:
            self.get_logger().error(f"Orchestration failed unexpectedly: {e}")
            self.speak("I'm having trouble with my thought process right now.")

    # ------------- Commands → action server -------------
    def _do_skill(self, text_lower: str, t0: float):
        t_skill0 = time.perf_counter()
        skill = "not_found"
        if re.search(r"\b(home|reset|park)\b", text_lower):  skill = "go_home"
        elif re.search(r"\b(open|release)\b", text_lower):    skill = "open_gripper"
        elif re.search(r"\b(close|grip|hold)\b", text_lower): skill = "close_gripper"

        if skill in self.allowed_skills:
            self.send_skill_goal(skill)
            self.speak(f"Executing {skill.replace('_',' ')}.")
        else:
            self.speak("I'm sorry, I don't know how to perform that action.")

        if TIMING_ENABLED:
            t_end = time.perf_counter()
            self._print_timing(
                stage="COMMAND",
                end_to_end_s=t_end - t0,
                skill_dispatch_s=t_end - t_skill0,
            )

    # ------------- Questions → brain (LLM/RAG) -------------
    def _do_question(self, question: str, t0: float):
        self.get_logger().info("Querying brain at http://localhost:8000/query (LLM/RAG)…")
        t_http0 = time.perf_counter()
        timing_server = {}

        try:
            r = requests.post(
                RAG_URL,
                json={"question": question},
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            js = r.json()
            answer = js.get("answer", "") or ""
            timing_server = js.get("timing", {}) or {}

            if not any(c.isalpha() for c in answer):
                answer = "I couldn't find a clear answer to that."

            self.speak(answer)

        except requests.exceptions.ReadTimeout:
            self.get_logger().error(f"Brain timeout after {HTTP_TIMEOUT:.1f}s.")
            self.speak("My thought process took too long and timed out.")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Brain connection error: {e}")
            self.speak("I can't connect to my knowledge base right now.")

        # timing printout
        if TIMING_ENABLED:
            t_end = time.perf_counter()
            self._print_timing(
                stage="QUESTION",
                end_to_end_s=t_end - t0,
                http_roundtrip_s=t_end - t_http0,
                server_total_s=timing_server.get("total_s"),
                server_retrieve_s=timing_server.get("retrieve_s"),
                server_generate_s=timing_server.get("generate_s"),
                mode=js.get("mode") if 'js' in locals() else None,
            )

    # ------------- Helpers -------------
    def send_skill_goal(self, skill_name: str):
        goal = SkillExecution.Goal()
        goal.skill_name = skill_name
        self.get_logger().info(f"Sending goal '{skill_name}' to Skill Server…")
        self._action_client.wait_for_server(timeout_sec=3.0)
        self._action_client.send_goal_async(goal)

    def speak(self, text: str):
        clean = strip_markdown(text)
        self.get_logger().info(f'Requesting to speak: "{clean}"')
        if not self.tts_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("TTS service not available.")
            return
        req = Speak.Request()
        req.text_to_speak = clean
        self.tts_client.call_async(req)  # fire-and-forget is fine for UX

    def _print_timing(self, stage: str, **kwargs):
        # pretty, single-line timing block for your terminal logs
        parts = [f"[TIMING {stage}]"]
        def add(label, val, fmt="{:.3f}s"):
            if val is None:
                return
            try:
                parts.append(f"{label}={fmt.format(float(val))}")
            except Exception:
                parts.append(f"{label}={val}")

        add("end2end", kwargs.get("end_to_end_s"))
        add("http", kwargs.get("http_roundtrip_s"))
        add("server_total", kwargs.get("server_total_s"))
        add("retrieve", kwargs.get("server_retrieve_s"))
        add("generate", kwargs.get("server_generate_s"))
        if kwargs.get("skill_dispatch_s") is not None:
            add("skill_dispatch", kwargs.get("skill_dispatch_s"))
        if kwargs.get("mode"):
            parts.append(f"mode={kwargs['mode']}")
        self.get_logger().info(" ".join(parts))

def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
