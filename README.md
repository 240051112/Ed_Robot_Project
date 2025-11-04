# Ed: An Offline Edge-AI Robotic Assistant for Smart Manufacturing

**My Author:** AGBOR Edouard Ransome (agboredouard51@gmail.com)
**My Supervisor:** Dr. Abdullah Alhusin Alkhdur
**Project:** MSc Smart Manufacturing, Aston University (2024/2025)

---

![My Ed Robot Setup] (images/my_robot_setup.png)
*(A photo of my final hardware setup, running on the NVIDIA Jetson Orin NX 16GB)*

### 1. Project Overview

This repository contains the complete source code for my dissertation project, **"Ed"**—a fully offline, edge-deployed robotic AI assistant we built for smart manufacturing environments.

Our work was motivated by a key industrial challenge: cloud-based AI is often too slow (high latency) and poses significant data privacy risks for factory floors. My solution, "Ed," runs **100% offline** on a single NVIDIA Jetson Orin NX 16GB, ensuring real-time responses and total data sovereignty.

### 2. Key Features

* **Sense-Think-Act Architecture:** A modular ROS 2 design that integrates perception (Sense), AI reasoning (Think), and robotic control (Act).
* **Dual-Path Language System:**
    * **Fast Path (LLM-only):** For general conversation and quick commands, providing rapid responses (p90 < 6s).
    * **Grounded Path (LLM+RAG):** Uses Retrieval-Augmented Generation to answer complex, factual questions by pulling data from local documents (like safety manuals, ISO standards, and technical specs).
* **Robust Safety & Persona:** The "Ed" persona I created is stable and enforced a **100% refusal rate** on unsafe or out-of-scope commands during testing.
* **ROS 2 Skill Server:** My AI brain calls robotic actions (like `center_tag`, `go_home`, `open_gripper`) as modular ROS 2 skills.

### 3. Hardware & Software Stack

| Component | Specification |
| :--- | :--- |
| **Compute** | NVIDIA Jetson Orin NX (16GB) |
| **Robotic Arm** | Yahboom DOFBOT Pro (6-DOF Arm, used as 5+1 DOF) |
| **Vision** | Orbbec RGB-D Camera (for AprilTag detection) |
| **Audio** | Jabra 510 Speaker/Microphone |
| **Middleware** | ROS 2 Humble |
| **AI Brain** | Phi-3-mini (Q4_K_M GGUF) via `llama-cpp-python` |
| **RAG** | Local ChromaDB Vector Store |
| **Voice I/O** | Vosk (ASR) & Piper (TTS) |

### 4. Key Performance Results

we validated the system using a 25-question evaluation suite covering safety, RAG accuracy, and performance.

* **Energy:** The system is highly efficient, running at a stable **~21.8W average / ~24.8W peak** power, proving it's feasible for all-day deployment.
* **Latency (LLM-only):** Succeeded in meeting my conversational target of < 6 seconds.
* **Latency (LLM+RAG):** Slower (~15s mean), which proved that my dual-path router (using the fast path by default) is an essential design choice.
* **Robotic Skill:** our `center_tag` visual servoing skill achieved a **9/10 success rate** in trials.
* **Cost & Sustainability:** My on-device solution is **~150-200x cheaper** per-query than a cloud API and uses **zero facility water**.

### 5. How to Use This Repository

The project is structured into several ROS 2 workspaces and setup scripts.

1.  **Workspaces:**
    * `ai_ed_ws`: The core AI brain (`ed_core`), voice I/O (`ed_io_voice`), and skills (`ed_skills`).
    * `drivers_ws`: Camera drivers.
    * `dofbot_pro_ws`: Robotic arm drivers, MoveIt configuration, and perception.
2.  **Setup Scripts:**
    * `setup_environment.sh`: Prepares the system.
    * `build_project.sh`: Cleans and builds all ROS 2 workspaces.
    * `run_ed_brain_offline.sh`: Launches the AI brain.

### 6. Project Demo

Here are some videos of the "Ed" assistant in action.

* [Video: Full System Demo](videos/full_system_demo.mkv)
* [Video: RAG in Action - Answering a Safety Question](videos/rag_safety_demo.mkv)

### 7. Acknowledgements

This project would not have been possible without the generous support of the **Chevening Scholarship** and the UK Government's Foreign, Commonwealth & Development Office (FCDO).

I also want to extend my sincere thanks to my supervisor, **Dr. Abdullah Alhusin Alkhdur**, for his invaluable guidance, and to all the academic staff at Aston University.