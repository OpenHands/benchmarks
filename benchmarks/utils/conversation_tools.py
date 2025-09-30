from openhands.sdk import get_logger
from openhands.sdk import Conversation, get_logger


logger = get_logger(__name__)

def get_history(conversation):
    # Check conversation state
    logger.info(f"Conversation status: {conversation.state.agent_status}")
    logger.info(f"Number of events: {len(list(conversation.state.events))}")

    # Extract conversation history
    logger.info(f"DEBUG: Conversation type: {type(conversation)}")
    logger.info(f"DEBUG: Conversation state type: {type(conversation.state)}")
    logger.info(f"DEBUG: Has events attribute: {hasattr(conversation.state, 'events')}")
    
    try:
        # For remote conversations, try to force a sync first
        if hasattr(conversation.state, 'events') and hasattr(conversation.state.events, '_do_full_sync'):
            logger.info("DEBUG: Forcing full sync for remote events...")
            conversation.state.events._do_full_sync()
        
        history = list(conversation.state.events)
        logger.info(f"Extracted {len(history)} events from conversation history")
        
        # Log some details about the events
        if history:
            logger.info(f"DEBUG: First event type: {type(history[0])}")
            logger.info(f"DEBUG: Last event type: {type(history[-1])}")
        else:
            logger.warning("DEBUG: No events found in conversation.state.events")
            
    except Exception as e:
        logger.error(f"Error extracting conversation history: {e}")
        logger.info(f"DEBUG: Trying alternative history extraction methods...")
        
        # Try alternative methods to get conversation history
        history = []
        if hasattr(conversation, '_events'):
            history = list(conversation._events)
            logger.info(f"Found {len(history)} events in conversation._events")
        elif hasattr(conversation, 'events'):
            history = list(conversation.events)
            logger.info(f"Found {len(history)} events in conversation.events")
        elif hasattr(conversation.state, '_events'):
            history = list(conversation.state._events)
            logger.info(f"Found {len(history)} events in conversation.state._events")
        else:
            logger.error("No events found in conversation object")
            # Try to inspect the conversation object
            logger.info(f"DEBUG: Conversation attributes: {dir(conversation)}")
            logger.info(f"DEBUG: Conversation state attributes: {dir(conversation.state)}")
            history = []
    return history

def get_git_patch_from_history(history):
    # Extract git patch from conversation history
    git_patch = ""
    workspace_path = None
    
    try:
        # Look for workspace path and any git diff output in conversation events
        import re
        logger.info(f"DEBUG: Analyzing {len(history)} events for git patches...")
        
        for i, event in enumerate(history):
            event_type = type(event).__name__
            logger.info(f"DEBUG: Event {i}: {event_type}")
            
            # Check different event attributes for content
            content_sources = []
            if hasattr(event, 'content'):
                content_sources.append(('content', event.content))
            if hasattr(event, 'observation'):
                content_sources.append(('observation', event.observation))
            if hasattr(event, 'action') and hasattr(event.action, 'content'):
                content_sources.append(('action.content', event.action.content))
            if hasattr(event, 'action') and hasattr(event.action, 'command'):
                content_sources.append(('action.command', event.action.command))
            
            for source_name, content in content_sources:
                if isinstance(content, str) and content:
                    logger.info(f"DEBUG: Event {i} {source_name} length: {len(content)}")
                    
                    # Extract workspace path if not found yet
                    if workspace_path is None and '/tmp/tmp' in content:
                        match = re.search(r'/tmp/tmp\w+/\w+', content)
                        if match:
                            workspace_path = match.group(0)
                            logger.info(f"Found workspace path: {workspace_path}")
                    
                    # Look for git diff output in event content
                    if ('diff --git' in content or 
                        ('--- a/' in content and '+++ b/' in content) or
                        (content.startswith('diff ') and '@@' in content)):
                        git_patch = content
                        logger.info(f"Found git patch in {source_name}: {len(git_patch)} characters")
                        logger.info(f"Git patch preview: {git_patch[:200]}...")
                        break
                    
                    # Also look for git commands that might produce diffs
                    if 'git diff' in content.lower():
                        logger.info(f"Found 'git diff' command in {source_name}: {content[:100]}...")
            
            # Also check if event has action with path
            if hasattr(event, 'action') and hasattr(event.action, 'path'):
                if workspace_path is None and '/tmp/tmp' in str(event.action.path):
                    match = re.search(r'/tmp/tmp\w+/\w+', str(event.action.path))
                    if match:
                        workspace_path = match.group(0)
                        logger.info(f"Found workspace path from action: {workspace_path}")
        
        # If no git patch found in history but we have workspace path, 
        # assume there were no changes (empty patch)
        if not git_patch and workspace_path:
            logger.info("No git patch found in conversation history - assuming no changes made")
            git_patch = ""
            
    except Exception as e:
        logger.error(f"Error extracting git patch: {e}")
        git_patch = ""
    return git_patch

