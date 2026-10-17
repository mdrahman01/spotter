#!/bin/sh
# Install the pre-commit hook that runs scripts/check_secrets.py on the staged diff before every commit.
#
#     sh scripts/install_hooks.sh
#
# The hook is two lines of sh. Git does not track hooks, so each clone runs this once. An older hook
# that is not ours is kept next to it as pre-commit.old. To forbid extra words, such as the owner's
# plate, add --word WORD to the python3 line of the hook, which stays out of the repository.
set -eu
cd "$(dirname "$0")/.."
hook="$(git rev-parse --git-path hooks/pre-commit)"
mkdir -p "$(dirname "$hook")"
if [ -f "$hook" ] && ! grep -q 'check_secrets.py' "$hook"; then
    cp "$hook" "$hook.old"
    echo "kept the old hook as $hook.old"
fi
printf '%s\n' '#!/bin/sh' 'cd "$(git rev-parse --show-toplevel)" && exec python3 scripts/check_secrets.py' > "$hook"
chmod +x "$hook"
echo "wrote $hook (executable): every commit now runs python3 scripts/check_secrets.py on the staged diff"
echo "try it on the whole repository: python3 scripts/check_secrets.py --all"
