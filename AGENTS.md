# Corgi RL

- Keep all repository documentation, comments, diagrams, captions and user-facing text in English.
- Python environments, dependency installation, simulation, training and tests run on the remote NVIDIA machine. Do not install or run the Python project on the owner's Mac.
- Keep private access instructions, credentials, hostnames and IP addresses out of this public repository.
- Use the shared specification in `src/corgi_rl/specs/corgi.json`. Generate MJCF from it; do not hand-edit generated models.
- The starting dimensions, masses and actuator parameters are estimates. Do not present them as supplier specifications.
- Report actual checks, device, steps and limitations. A smoke test is not convergence or hardware readiness.
- Preserve the observation/action order and specification hash in every exported policy.
