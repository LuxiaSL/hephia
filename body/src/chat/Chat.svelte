<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { setupIPCListeners } from '$state/ipc-listener';
  import { chatMessages, addChatMessage, addChatResponse } from '$state/stores';
  import { sendChatMessage } from '$lib/tauri-commands';
  import type { ChatMessage, ChatResponse } from '$lib/types';

  let messages: ChatMessage[] = [];
  let inputText = '';
  let sending = false;
  let messagesContainer: HTMLDivElement;
  let unlisten: (() => void) | null = null;

  chatMessages.subscribe((v) => {
    messages = v;
    // Auto-scroll to bottom on new message
    requestAnimationFrame(() => {
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    });
  });

  onMount(async () => {
    unlisten = await setupIPCListeners({
      onChat: (response: ChatResponse) => {
        addChatResponse(response);
        sending = false;
      },
    });
  });

  onDestroy(() => {
    if (unlisten) unlisten();
  });

  async function handleSend() {
    const text = inputText.trim();
    if (!text || sending) return;

    inputText = '';
    sending = true;

    addChatMessage({ role: 'user', content: text });

    try {
      await sendChatMessage(text);
    } catch (e) {
      addChatMessage({
        role: 'system',
        content: `Failed to send: ${e}`,
      });
      sending = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleClose() {
    const win = getCurrentWindow();
    await win.hide();
  }
</script>

<div class="chat-root">
  <!-- Custom title bar -->
  <div class="titlebar">
    <span class="titlebar-title">Hephia Chat</span>
    <div class="titlebar-controls">
      <button class="titlebar-btn close" on:click={handleClose}>✕</button>
    </div>
  </div>

  <!-- Messages -->
  <div class="messages" bind:this={messagesContainer}>
    {#each messages as msg}
      <div class="message message-{msg.role}">
        <div class="message-content">{msg.content}</div>
      </div>
    {/each}
    {#if sending}
      <div class="message message-assistant typing">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    {/if}
  </div>

  <!-- Input -->
  <div class="input-area">
    <input
      type="text"
      bind:value={inputText}
      on:keydown={handleKeydown}
      placeholder="Type a message..."
      disabled={sending}
    />
    <button class="btn btn-primary send-btn" on:click={handleSend} disabled={sending || !inputText.trim()}>
      Send
    </button>
  </div>
</div>

<style>
  .chat-root {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 32px;
    padding: 0 8px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    -webkit-app-region: drag;
    flex-shrink: 0;
  }

  .titlebar-title {
    font-size: 12px;
    color: var(--text-secondary);
    pointer-events: none;
  }

  .titlebar-controls {
    -webkit-app-region: no-drag;
  }

  .titlebar-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
  }

  .titlebar-btn.close:hover {
    background: var(--error);
    color: white;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .message {
    max-width: 85%;
    padding: 8px 12px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
  }

  .message-user {
    align-self: flex-end;
    background: var(--accent-primary);
    color: white;
    border-bottom-right-radius: 4px;
  }

  .message-assistant {
    align-self: flex-start;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
  }

  .message-system {
    align-self: center;
    background: transparent;
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
  }

  .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 4px 0;
  }

  .typing-indicator span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    animation: typing 1.4s infinite ease-in-out;
  }

  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-4px); }
  }

  .input-area {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
    flex-shrink: 0;
  }

  .input-area input {
    flex: 1;
  }

  .send-btn {
    flex-shrink: 0;
    padding: 8px 16px;
  }
</style>
