from setuptools import setup, find_packages
from pathlib import Path

package_name = "ed_core"
root = Path(__file__).parent

def collect_docs():
    docs = []
    docs_dir = root / "documents"
    if docs_dir.exists():
        for pat in ("**/*.pdf", "**/*.txt", "**/*.md"):
            for p in docs_dir.rglob(pat):
                if p.is_file():
                    docs.append(str(p.relative_to(root)))
    return docs

setup(
    name=package_name,
    version="0.3.4",  # keep in sync with package.xml
    packages=find_packages(include=["ed_core", "ed_core.*"]),
    package_data={"ed_core.rag_service": ["system_prompt.md"]},
    include_package_data=True,

    # Ensure ROS resource index + package.xml + docs + scripts are installed
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/documents", collect_docs()),
        # Install robust wrappers regardless of entry_points handling
        (f"lib/{package_name}", [
            "scripts/ed_brain",
            "scripts/run_ed_brain",
            "scripts/ed_brain_tmux",
        ]),
    ],

    install_requires=[
        "fastapi>=0.110",
        "uvicorn>=0.23",
        "scikit-learn>=1.1",
        "numpy>=1.23",
    ],
    zip_safe=False,
    maintainer="jetson",
    maintainer_email="jetson@todo.todo",
    description="The RAG/LLM core brain for the Ed robot.",
    license="Apache-2.0",

    # Also create console_script entry points (redundant safety net)
    entry_points={
        "console_scripts": [
            "ed_brain = ed_core.rag_service.api:main",
            "run_ed_brain = ed_core.rag_service.api:main",
        ],
    },
)
