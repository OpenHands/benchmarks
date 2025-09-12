checkout_eval_branch() {
    if [ -z "$COMMIT_HASH" ]; then
        echo "Commit hash not specified, use current git commit"
        return 0
    fi

    if git diff --quiet $COMMIT_HASH HEAD; then
        echo "The given hash is equivalent to the current HEAD"
        return 0
    fi

    echo "Start to checkout openhands version to $COMMIT_HASH, but keep current evaluation harness"
    if ! git diff-index --quiet HEAD --; then
        echo "There are uncommitted changes, please stash or commit them first"
        exit 1
    fi
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo "Current version is: $current_branch"
    echo "Check out OpenHands to version: $COMMIT_HASH"
    if ! git checkout $COMMIT_HASH; then
        echo "Failed to check out to $COMMIT_HASH"
        exit 1
    fi

    echo "Revert changes in evaluation folder"
    git checkout $current_branch -- evaluation

    # Trap the EXIT signal to checkout original branch
    trap checkout_original_branch EXIT

}


checkout_original_branch() {
    if [ -z "$current_branch" ]; then
        return 0
    fi
    echo "Checkout back to original branch $current_branch"
    git checkout $current_branch
}

get_openhands_version() {
    # IMPORTANT: Because Agent's prompt changes fairly often in the rapidly evolving codebase of OpenHands
    # We need to track the version of Agent in the evaluation to make sure results are comparable
    if [ -z "$OPENHANDS_SDK" ]; then
        echo "Warning: OPENHANDS_SDK environment variable not set, using default version" >&2
        OPENHANDS_VERSION="v1-unknown"
        return
    fi
    
    # Try to get version from pyproject.toml in the SDK subdirectory
    if [ -f "$OPENHANDS_SDK/openhands/sdk/pyproject.toml" ]; then
        VERSION=$(grep -E '^version\s*=' "$OPENHANDS_SDK/openhands/sdk/pyproject.toml" | sed -E 's/.*version\s*=\s*"([^"]+)".*/\1/')
        if [ -n "$VERSION" ]; then
            OPENHANDS_VERSION="v$VERSION"
            return
        fi
    fi
    
    # Fallback: try to get version from root pyproject.toml (older format)
    if [ -f "$OPENHANDS_SDK/pyproject.toml" ]; then
        VERSION=$(grep -E '^version\s*=' "$OPENHANDS_SDK/pyproject.toml" | sed -E 's/.*version\s*=\s*"([^"]+)".*/\1/')
        if [ -n "$VERSION" ]; then
            OPENHANDS_VERSION="v$VERSION"
            return
        fi
    fi
    
    # Fallback: try to get git commit hash from SDK directory
    if [ -d "$OPENHANDS_SDK/.git" ]; then
        cd "$OPENHANDS_SDK"
        GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null)
        if [ -n "$GIT_HASH" ]; then
            OPENHANDS_VERSION="v1-$GIT_HASH"
            cd - > /dev/null
            return
        fi
        cd - > /dev/null
    fi
    
    # Final fallback
    OPENHANDS_VERSION="v1-sdk"
}

# If script is executed directly (not sourced), run the function and print the version
# Check if script is being executed directly by looking at $0
case "$0" in
    *version_control.sh)
        # Script is being executed directly
        get_openhands_version
        echo "$OPENHANDS_VERSION"
        ;;
    *)
        # Script is being sourced, do nothing
        :
        ;;
esac
