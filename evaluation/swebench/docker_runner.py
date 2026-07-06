"""Run misterdev on a SWE-bench instance inside its official Docker image.

The blocker for a real run is the per-repo environment (exact dependency
versions). SWE-bench ships that as a Docker image per instance. misterdev already
has a container execution env — it bind-mounts the working tree into an image and
runs the build/test gates inside it — so the clean integration is: misterdev
edits files on the HOST, but its gates run INSIDE the SWE-bench image, mounted at
/testbed (where that image's editable install resolves). No misterdev install in
the container is needed.

This module derives the image name, writes the container-env project.yaml, and
grades the patch inside the image. Building the images and driving the suite is
left to the caller / CLI because it is slow (emulated on Apple Silicon) and
costs model budget.
"""

import subprocess
from pathlib import Path
from typing import List

from .grader import GradeResult, apply_patch
from .instance import SWEBenchInstance


def instance_image(instance: SWEBenchInstance, arch: str = "x86_64") -> str:
    """The official SWE-bench evaluation image tag for an instance."""
    return f"sweb.eval.{arch}.{instance.instance_id}:latest"


def build_images(instances: List[SWEBenchInstance], max_workers: int = 4) -> None:
    """Build the SWE-bench evaluation images for ``instances`` (via the official
    harness). Slow and disk-heavy; run once, then reuse the cached images."""
    from swebench.harness.docker_build import build_instance_images
    from swebench.harness.test_spec.test_spec import make_test_spec
    import docker

    client = docker.from_env()
    specs = [make_test_spec({"instance_id": i.instance_id}) for i in instances]
    build_instance_images(client=client, dataset=specs, max_workers=max_workers)


def write_container_project_yaml(
    repo: Path, instance: SWEBenchInstance, arch: str = "x86_64"
) -> None:
    """Write a project.yaml that routes misterdev's gates through the instance
    image, mounted at /testbed so the image's editable install sees the edits."""
    cfg = (
        f'name: "{instance.instance_id}"\n'
        f'language: "{instance.language}"\n'
        f'test_command: "{instance.test_command}"\n'
        "environment:\n"
        "  type: docker\n"
        f'  image: "{instance_image(instance, arch)}"\n'
        '  mount_path: "/testbed"\n'
    )
    (repo / "project.yaml").write_text(cfg, encoding="utf-8")


def _docker_test(image: str, host_repo: Path, command: str, timeout: int) -> str:
    """Run a test command against the host repo bind-mounted into the image."""
    argv = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{host_repo}:/testbed",
        "-w",
        "/testbed",
        image,
        "sh",
        "-c",
        command,
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def grade_in_container(
    repo: str,
    instance: SWEBenchInstance,
    arch: str = "x86_64",
    timeout: int = 1800,
) -> GradeResult:
    """Grade the host working tree (model patch applied) inside the instance image.

    Applies the task's test_patch on the host tree, then runs FAIL_TO_PASS and
    PASS_TO_PASS INSIDE the image (where the deps are) against the bind-mounted
    tree. Resolved only when every FAIL_TO_PASS passes and PASS_TO_PASS holds.
    """
    from .grader import _SUMMARY_LINE, _PASSING

    root = Path(repo)
    if not instance.fail_to_pass:
        return GradeResult(False, error="instance has no FAIL_TO_PASS tests")
    if not apply_patch(root, instance.test_patch):
        return GradeResult(False, error="test_patch did not apply")

    all_ids = list(instance.fail_to_pass) + list(instance.pass_to_pass)
    quoted = " ".join(f"'{n}'" for n in all_ids)
    output = _docker_test(
        instance_image(instance, arch),
        root,
        f"{instance.test_command} {quoted}",
        timeout,
    )
    outcomes = {n: False for n in all_ids}
    for line in output.splitlines():
        m = _SUMMARY_LINE.match(line.strip())
        if m and m.group(2) in outcomes:
            outcomes[m.group(2)] = m.group(1) in _PASSING
    ftp = {n: outcomes[n] for n in instance.fail_to_pass}
    ptp = {n: outcomes[n] for n in instance.pass_to_pass}
    return GradeResult(
        resolved=all(ftp.values()) and all(ptp.values()),
        fail_to_pass=ftp,
        pass_to_pass=ptp,
    )
