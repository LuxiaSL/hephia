<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { setupIPCListeners } from '$state/ipc-listener';
  import { systemContext } from '$state/stores';
  import { marked } from 'marked';
  import {
    getActions,
    performAction,
    getSettings,
    updateSettings,
    getStateSnapshot,
    submitWorkerTask,
    getWorkerStatus,
    replyToWorkerTask,
    stopWorkerTask,
    deleteWorkerTask,
    clearWorkerTasks,
    starWorkerTask,
  } from '$lib/tauri-commands';
  import type {
    SystemContext,
    SoulStatePayload,
    ActionInfo,
    AgentQuestion,
    PetSettings,
    WorkerTaskStatus,
  } from '$lib/types';

  // Configure marked for safe rendering
  marked.setOptions({
    breaks: true,
    gfm: true,
  });

  function renderMarkdown(text: string): string {
    try {
      return marked.parse(text) as string;
    } catch {
      return text;
    }
  }

  let activeTab = 'state';
  let context: SystemContext | null = null;
  let unlisten: (() => void) | null = null;

  // Actions tab state
  let actions: Record<string, ActionInfo> = {};
  let actionsLoading = false;
  let actionsError = '';
  let executingAction = '';

  // Worker tab state
  let workerInput = '';
  let workerSubmitting = false;
  let workerError = '';
  let workerTasks: WorkerTaskStatus[] = [];
  let workerPollTimers: Map<string, number> = new Map();
  let expandedTask: string | null = null;
  let replyInputs: Map<string, string> = new Map();
  let replyingTask: string | null = null;
  let stoppingTask: string | null = null;
  let deletingTask: string | null = null;
  let starredTasks: Set<string> = new Set();
  let starringTask: string | null = null;
  let clearingTasks = false;

  // Multi-question selection: taskId → questionIndex → selected label
  let selectedAnswers: Map<string, Map<number, string>> = new Map();

  // Progress feed auto-scroll
  let progressFeedEl: HTMLDivElement | null = null;
  let userScrolledUp = false;
  let lastProgressCount = 0;

  // Settings tab state
  let settings: PetSettings | null = null;
  let availableModels: string[] = [];
  let settingsLoading = false;
  let settingsError = '';
  let settingsSaving = false;
  let settingsSuccess = false;

  const tabs = [
    { id: 'state', label: 'State' },
    { id: 'actions', label: 'Actions' },
    { id: 'worker', label: 'Worker' },
    { id: 'settings', label: 'Settings' },
  ];

  /** Ensure all SystemContext fields are present with safe defaults. */
  function normalizeContext(raw: Record<string, any>): SystemContext {
    return {
      mood: raw.mood ?? { name: 'neutral', valence: 0, arousal: 0 },
      needs: raw.needs ?? {},
      behavior: raw.behavior ?? { name: 'idle' },
      emotional_state: raw.emotional_state ?? [],
    };
  }

  systemContext.subscribe((v) => (context = v));

  onMount(async () => {
    unlisten = await setupIPCListeners({
      onState: (payload: SoulStatePayload) => {
        const ctx = payload.payload?.system_context;
        if (ctx) systemContext.set(normalizeContext(ctx));
      },
    });

    // Fetch initial state immediately rather than waiting for next broadcast
    try {
      const snapshot = await getStateSnapshot();
      if (snapshot) {
        systemContext.set(normalizeContext(snapshot as Record<string, any>));
      }
    } catch {
      // Backend might not be ready yet — IPC listener will pick up state later
    }
  });

  onDestroy(() => {
    if (unlisten) unlisten();
  });

  async function handleClose() {
    const win = getCurrentWindow();
    await win.hide();
  }

  function handleTitlebarDrag() {
    getCurrentWindow().startDragging();
  }

  async function switchTab(tabId: string) {
    activeTab = tabId;
    if (tabId === 'actions' && Object.keys(actions).length === 0) {
      await loadActions();
    }
    if (tabId === 'settings' && !settings) {
      await loadSettings();
    }
  }

  // --- Actions ---

  async function loadActions() {
    actionsLoading = true;
    actionsError = '';
    try {
      actions = await getActions();
    } catch (e) {
      actionsError = `Failed to load actions: ${e}`;
    }
    actionsLoading = false;
  }

  async function executeAction(name: string) {
    executingAction = name;
    try {
      const result = await performAction(name);
      if (!result.success) {
        actionsError = result.message;
      }
      // Refresh actions to update cooldowns
      await loadActions();
    } catch (e) {
      actionsError = `Action failed: ${e}`;
    }
    executingAction = '';
  }

  // --- Settings ---

  async function loadSettings() {
    settingsLoading = true;
    settingsError = '';
    try {
      const resp = await getSettings();
      settings = { ...resp.settings };
      availableModels = resp.available_models;
    } catch (e) {
      settingsError = `Failed to load settings: ${e}`;
    }
    settingsLoading = false;
  }

  async function saveSettings() {
    if (!settings) return;
    settingsSaving = true;
    settingsError = '';
    settingsSuccess = false;
    try {
      const resp = await updateSettings(settings);
      settings = { ...resp.settings };
      settingsSuccess = true;
      setTimeout(() => (settingsSuccess = false), 2000);
    } catch (e) {
      settingsError = `Failed to save: ${e}`;
    }
    settingsSaving = false;
  }

  // --- Worker ---

  async function handleWorkerSubmit() {
    const task = workerInput.trim();
    if (!task) return;

    workerSubmitting = true;
    workerError = '';
    try {
      console.log('[Worker] Submitting task:', task);
      const resp = await submitWorkerTask(task);
      console.log('[Worker] Submit response:', resp);
      workerInput = '';

      // Create initial status entry
      const status: WorkerTaskStatus = {
        task_id: resp.task_id,
        status: 'pending',
        result: null,
        error: null,
        created_at: Date.now() / 1000,
        completed_at: null,
        progress: [],
        cost_usd: null,
        num_turns: null,
        pending_questions: [],
      };
      workerTasks = [status, ...workerTasks];

      // Start polling for this task
      startPolling(resp.task_id);
    } catch (e) {
      console.error('[Worker] Submit failed:', e);
      workerError = `Failed to submit task: ${e}`;
    }
    workerSubmitting = false;
  }

  function startPolling(taskId: string) {
    // Don't double-poll
    if (workerPollTimers.has(taskId)) return;
    console.log('[Worker] Starting poll for:', taskId);
    const timer = window.setInterval(async () => {
      try {
        const status = await getWorkerStatus(taskId);
        console.log('[Worker] Poll result:', taskId.slice(0, 8), status.status, `(${status.progress?.length ?? 0} steps)`);
        workerTasks = workerTasks.map((t) =>
          t.task_id === taskId ? status : t,
        );

        // Auto-scroll progress feed if new entries arrived for the expanded task
        if (taskId === expandedTask && status.progress.length > lastProgressCount) {
          lastProgressCount = status.progress.length;
          requestAnimationFrame(autoScrollProgress);
        }

        // Stop polling when task is done (but NOT on awaiting_input — keep polling)
        if (status.status === 'completed' || status.status === 'failed') {
          stopPolling(taskId);
        }
      } catch (e) {
        console.error('[Worker] Poll error for', taskId.slice(0, 8), ':', e);
        stopPolling(taskId);
      }
    }, 2000);
    workerPollTimers.set(taskId, timer);
  }

  function stopPolling(taskId: string) {
    const timer = workerPollTimers.get(taskId);
    if (timer !== undefined) {
      window.clearInterval(timer);
      workerPollTimers.delete(taskId);
    }
  }

  /** Unified reply handler — works for any task state. */
  async function handleReply(taskId: string, message?: string) {
    const msg = message ?? replyInputs.get(taskId)?.trim();
    if (!msg) return;

    replyingTask = taskId;
    workerError = '';
    try {
      // Optimistic update: show running, clear questions
      workerTasks = workerTasks.map((t) =>
        t.task_id === taskId
          ? { ...t, status: 'running' as const, pending_questions: [], completed_at: null }
          : t,
      );

      await replyToWorkerTask(taskId, msg);

      // Clear the input
      replyInputs.delete(taskId);
      replyInputs = replyInputs;

      // Ensure polling is active
      startPolling(taskId);
    } catch (e) {
      console.error('[Worker] Reply failed:', e);
      workerError = `Reply failed: ${e}`;
    }
    replyingTask = null;
  }

  async function handleStop(taskId: string) {
    stoppingTask = taskId;
    try {
      const status = await stopWorkerTask(taskId);
      workerTasks = workerTasks.map((t) =>
        t.task_id === taskId ? status : t,
      );
      stopPolling(taskId);
    } catch (e) {
      console.error('[Worker] Stop failed:', e);
      workerError = `Stop failed: ${e}`;
    }
    stoppingTask = null;
  }

  async function handleDelete(taskId: string) {
    deletingTask = taskId;
    try {
      await deleteWorkerTask(taskId);
      // Optimistic removal
      workerTasks = workerTasks.filter((t) => t.task_id !== taskId);
      stopPolling(taskId);
      if (expandedTask === taskId) expandedTask = null;
    } catch (e) {
      console.error('[Worker] Delete failed:', e);
      workerError = `Delete failed: ${e}`;
    }
    deletingTask = null;
  }

  async function handleClearCompleted() {
    clearingTasks = true;
    workerError = '';
    try {
      await clearWorkerTasks(true);
      // Remove completed/failed tasks from local array
      workerTasks = workerTasks.filter(
        (t) => t.status !== 'completed' && t.status !== 'failed',
      );
      if (expandedTask && !workerTasks.find((t) => t.task_id === expandedTask)) {
        expandedTask = null;
      }
    } catch (e) {
      console.error('[Worker] Clear failed:', e);
      workerError = `Clear failed: ${e}`;
    }
    clearingTasks = false;
  }

  async function handleStar(taskId: string) {
    starringTask = taskId;
    try {
      const resp = await starWorkerTask(taskId);
      if (resp.node_id) {
        starredTasks.add(taskId);
        starredTasks = starredTasks; // trigger reactivity
      } else if (resp.error) {
        workerError = `Star failed: ${resp.error}`;
      }
    } catch (e) {
      console.error('[Worker] Star failed:', e);
      workerError = `Star failed: ${e}`;
    }
    starringTask = null;
  }

  $: hasCompletedTasks = workerTasks.some(
    (t) => t.status === 'completed' || t.status === 'failed',
  );

  function selectAnswer(taskId: string, questionIndex: number, label: string) {
    if (!selectedAnswers.has(taskId)) {
      selectedAnswers.set(taskId, new Map());
    }
    const taskAnswers = selectedAnswers.get(taskId)!;
    // Toggle: deselect if already selected
    if (taskAnswers.get(questionIndex) === label) {
      taskAnswers.delete(questionIndex);
    } else {
      taskAnswers.set(questionIndex, label);
    }
    selectedAnswers = selectedAnswers; // trigger reactivity
  }

  function getSelectedAnswer(taskId: string, questionIndex: number): string | undefined {
    return selectedAnswers.get(taskId)?.get(questionIndex);
  }

  function hasAllAnswers(taskId: string, questionCount: number): boolean {
    const taskAnswers = selectedAnswers.get(taskId);
    if (!taskAnswers) return false;
    return taskAnswers.size >= questionCount;
  }

  function hasAnyAnswer(taskId: string): boolean {
    const taskAnswers = selectedAnswers.get(taskId);
    return !!taskAnswers && taskAnswers.size > 0;
  }

  async function sendSelectedAnswers(taskId: string, questions: AgentQuestion[]) {
    const taskAnswers = selectedAnswers.get(taskId);
    if (!taskAnswers || taskAnswers.size === 0) return;

    // Build reply: if single question, just send the label; if multiple, format as numbered list
    let msg: string;
    if (questions.length === 1) {
      msg = taskAnswers.get(0) ?? '';
    } else {
      const parts: string[] = [];
      for (let i = 0; i < questions.length; i++) {
        const answer = taskAnswers.get(i);
        if (answer) {
          parts.push(`${i + 1}. ${answer}`);
        }
      }
      msg = parts.join('\n');
    }

    // Clear selections and send
    selectedAnswers.delete(taskId);
    selectedAnswers = selectedAnswers;
    await handleReply(taskId, msg);
  }

  function getReplyInput(taskId: string): string {
    return replyInputs.get(taskId) ?? '';
  }

  function setReplyInput(taskId: string, value: string) {
    replyInputs.set(taskId, value);
    replyInputs = replyInputs; // trigger reactivity
  }

  function toggleExpand(taskId: string) {
    expandedTask = expandedTask === taskId ? null : taskId;
    // Reset scroll state when expanding a new task
    userScrolledUp = false;
    lastProgressCount = 0;
  }

  function handleProgressScroll(e: Event) {
    const el = e.target as HTMLDivElement;
    if (!el) return;
    // Consider "at bottom" if within 30px of the end
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    userScrolledUp = !atBottom;
  }

  function autoScrollProgress() {
    if (userScrolledUp || !progressFeedEl) return;
    progressFeedEl.scrollTop = progressFeedEl.scrollHeight;
  }

  function formatTime(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString();
  }

  // Clean up poll timers on destroy
  onDestroy(() => {
    for (const timer of workerPollTimers.values()) {
      window.clearInterval(timer);
    }
    workerPollTimers.clear();
  });
</script>

<div class="dashboard-root">
  <!-- Title bar -->
  <div class="titlebar" on:mousedown={handleTitlebarDrag}>
    <span class="titlebar-title">Hephia Dashboard</span>
    <div class="titlebar-controls">
      <button class="titlebar-btn close" on:click|stopPropagation={handleClose}>✕</button>
    </div>
  </div>

  <!-- Tab bar -->
  <div class="tab-bar">
    {#each tabs as tab}
      <button
        class="tab"
        class:active={activeTab === tab.id}
        on:click={() => switchTab(tab.id)}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <!-- Tab content -->
  <div class="tab-content">
    {#if activeTab === 'state'}
      <!-- STATE TAB -->
      <div class="tab-panel">
        {#if context}
          <div class="state-section">
            <h3>Mood</h3>
            <div class="mood-display">
              <span class="mood-name">{context.mood.name}</span>
              <span class="mood-values">V: {context.mood.valence.toFixed(2)} / A: {context.mood.arousal.toFixed(2)}</span>
            </div>
          </div>

          <div class="state-section">
            <h3>Needs</h3>
            {#each Object.entries(context.needs) as [name, need]}
              <div class="need-row">
                <span class="need-name">{name}</span>
                <div class="need-bar">
                  <div
                    class="need-fill"
                    class:low={need.satisfaction < 0.3}
                    class:mid={need.satisfaction >= 0.3 && need.satisfaction < 0.7}
                    class:high={need.satisfaction >= 0.7}
                    style="width: {need.satisfaction * 100}%"
                  ></div>
                </div>
                <span class="need-value">{Math.round(need.satisfaction * 100)}%</span>
              </div>
            {/each}
          </div>

          <div class="state-section">
            <h3>Behavior</h3>
            <span class="behavior-name">{context.behavior.name ?? 'none'}</span>
          </div>

          <div class="state-section">
            <h3>Emotions</h3>
            {#if context.emotional_state.length === 0}
              <span class="empty">No active emotions</span>
            {:else}
              {#each context.emotional_state as emotion}
                <div class="emotion-row">
                  <span class="emotion-name">{emotion.name}</span>
                  <div class="emotion-bar">
                    <div class="emotion-fill" style="width: {emotion.intensity * 100}%"></div>
                  </div>
                  <span class="emotion-value">{emotion.intensity.toFixed(2)}</span>
                </div>
              {/each}
            {/if}
          </div>
        {:else}
          <div class="loading">Waiting for connection...</div>
        {/if}
      </div>

    {:else if activeTab === 'actions'}
      <!-- ACTIONS TAB -->
      <div class="tab-panel">
        {#if actionsLoading}
          <div class="loading">Loading actions...</div>
        {:else if actionsError}
          <div class="error-msg">{actionsError}</div>
        {:else if Object.keys(actions).length === 0}
          <div class="empty">No actions available</div>
        {:else}
          <div class="actions-list">
            {#each Object.entries(actions) as [name, action]}
              <div class="action-card">
                <div class="action-header">
                  <span class="action-name">{name}</span>
                  <button
                    class="btn btn-sm"
                    on:click={() => executeAction(name)}
                    disabled={executingAction === name || action.status.on_cooldown}
                  >
                    {#if executingAction === name}
                      Running...
                    {:else if action.status.on_cooldown}
                      Cooldown ({Math.round(action.status.remaining_cooldown)}s)
                    {:else}
                      Execute
                    {/if}
                  </button>
                </div>
                <div class="action-desc">{action.description}</div>
                <div class="action-stats">
                  Runs: {action.status.total_executions} |
                  Success: {action.status.successful_executions} |
                  Failed: {action.status.failed_executions}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

    {:else if activeTab === 'worker'}
      <!-- WORKER TAB -->
      <div class="tab-panel">
        <div class="worker-submit">
          <div class="worker-input-row">
            <input
              type="text"
              class="worker-input"
              bind:value={workerInput}
              placeholder="Describe a task..."
              on:keydown={(e) => { if (e.key === 'Enter' && !workerSubmitting) handleWorkerSubmit(); }}
              disabled={workerSubmitting}
            />
            <button
              class="btn btn-primary btn-sm"
              on:click={handleWorkerSubmit}
              disabled={workerSubmitting || !workerInput.trim()}
            >
              {workerSubmitting ? 'Submitting...' : 'Submit'}
            </button>
          </div>
          {#if workerError}
            <div class="error-msg">{workerError}</div>
          {/if}
        </div>

        {#if workerTasks.length === 0}
          <div class="empty">No tasks submitted yet. Describe something for the worker model to do.</div>
        {:else}
          {#if hasCompletedTasks}
            <div class="clear-row">
              <button
                class="btn-link"
                on:click={handleClearCompleted}
                disabled={clearingTasks}
              >
                {clearingTasks ? 'Clearing...' : 'Clear completed'}
              </button>
            </div>
          {/if}
          <div class="worker-tasks">
            {#each workerTasks as task}
              <div class="worker-task-card" class:expanded={expandedTask === task.task_id}>
                <div class="task-header" on:click={() => toggleExpand(task.task_id)}>
                  <div class="task-meta">
                    <span
                      class="task-status"
                      class:pending={task.status === 'pending'}
                      class:running={task.status === 'running'}
                      class:completed={task.status === 'completed'}
                      class:failed={task.status === 'failed'}
                      class:awaiting={task.status === 'awaiting_input'}
                    >
                      {task.status === 'awaiting_input' ? 'needs input' : task.status}
                    </span>
                    <span class="task-id">{task.task_id.slice(0, 8)}</span>
                  </div>
                  <div class="task-header-right">
                    {#if task.status === 'completed' && task.result && !starredTasks.has(task.task_id)}
                      <button
                        class="btn-icon btn-star"
                        on:click|stopPropagation={() => handleStar(task.task_id)}
                        disabled={starringTask === task.task_id}
                        title="Save result to memory"
                      >
                        {starringTask === task.task_id ? '...' : '\u2606'}
                      </button>
                    {/if}
                    {#if starredTasks.has(task.task_id)}
                      <span class="starred-badge" title="Memorized">Memorized</span>
                    {/if}
                    {#if task.status !== 'running' && task.status !== 'pending'}
                      <button
                        class="btn-icon btn-delete"
                        on:click|stopPropagation={() => handleDelete(task.task_id)}
                        disabled={deletingTask === task.task_id}
                        title="Delete task"
                      >
                        {deletingTask === task.task_id ? '...' : '\u2715'}
                      </button>
                    {/if}
                    <span class="task-time">{formatTime(task.created_at)}</span>
                  </div>
                </div>

                {#if expandedTask === task.task_id}
                  <div class="task-details">
                    <!-- Progress bar for active tasks -->
                    {#if task.status === 'pending' || task.status === 'running' || task.status === 'awaiting_input'}
                      <div class="task-progress">
                        <div class="progress-bar-track">
                          <div class="progress-bar-fill" class:indeterminate={task.status !== 'awaiting_input'}></div>
                        </div>
                        <span class="progress-label">
                          {#if task.status === 'pending'}
                            Queued...
                          {:else if task.status === 'awaiting_input'}
                            Waiting for your input
                          {:else}
                            Working... ({task.progress.length} steps)
                          {/if}
                        </span>
                      </div>
                    {/if}

                    <!-- Progress feed (always shown if entries exist) -->
                    {#if task.progress.length > 0}
                      <div class="progress-feed" bind:this={progressFeedEl} on:scroll={handleProgressScroll}>
                        {#each task.progress as entry}
                          <div class="progress-entry">
                            <span class="progress-type-badge" class:tool={entry.type === 'tool_use'} class:thinking={entry.type === 'thinking'} class:error={entry.type === 'error'}>
                              {entry.type === 'tool_use' && entry.tool_name ? entry.tool_name : entry.type}
                            </span>
                            <span class="progress-content">{entry.content}</span>
                          </div>
                        {/each}
                      </div>
                    {/if}

                    <!-- Question cards when awaiting_input -->
                    {#if task.status === 'awaiting_input' && task.pending_questions.length > 0}
                      {#each task.pending_questions as q, qi}
                        <div class="question-card">
                          <div class="question-text">{q.question}</div>
                          {#if q.options.length > 0}
                            <div class="question-options">
                              {#each q.options as opt}
                                {#if task.pending_questions.length === 1}
                                  <!-- Single question: click to send immediately -->
                                  <button
                                    class="btn btn-option"
                                    on:click={() => handleReply(task.task_id, opt.label)}
                                    disabled={replyingTask === task.task_id}
                                    title={opt.description}
                                  >
                                    <span class="option-label">{opt.label}</span>
                                    {#if opt.description}
                                      <span class="option-desc">{opt.description}</span>
                                    {/if}
                                  </button>
                                {:else}
                                  <!-- Multiple questions: select, then send together -->
                                  <button
                                    class="btn btn-option"
                                    class:selected={getSelectedAnswer(task.task_id, qi) === opt.label}
                                    on:click={() => selectAnswer(task.task_id, qi, opt.label)}
                                    disabled={replyingTask === task.task_id}
                                    title={opt.description}
                                  >
                                    <span class="option-label">{opt.label}</span>
                                    {#if opt.description}
                                      <span class="option-desc">{opt.description}</span>
                                    {/if}
                                  </button>
                                {/if}
                              {/each}
                            </div>
                          {/if}
                        </div>
                      {/each}
                      <!-- Send button for multi-question selection -->
                      {#if task.pending_questions.length > 1 && hasAnyAnswer(task.task_id)}
                        <div class="send-answers-row">
                          <button
                            class="btn btn-primary btn-sm"
                            on:click={() => sendSelectedAnswers(task.task_id, task.pending_questions)}
                            disabled={replyingTask === task.task_id || !hasAllAnswers(task.task_id, task.pending_questions.length)}
                          >
                            {#if replyingTask === task.task_id}
                              Sending...
                            {:else if hasAllAnswers(task.task_id, task.pending_questions.length)}
                              Send answers
                            {:else}
                              Answer all questions ({selectedAnswers.get(task.task_id)?.size ?? 0}/{task.pending_questions.length})
                            {/if}
                          </button>
                        </div>
                      {/if}
                    {/if}

                    <!-- Result display -->
                    {#if task.status === 'completed' && task.result}
                      <div class="task-result markdown-body">{@html renderMarkdown(task.result)}</div>
                    {/if}

                    <!-- Error display -->
                    {#if task.status === 'failed' && task.error}
                      <div class="task-error">{task.error}</div>
                    {/if}

                    <!-- Stats + completion time -->
                    {#if task.num_turns != null || task.cost_usd != null}
                      <div class="task-stats">
                        {#if task.num_turns != null}<span>{task.num_turns} turns</span>{/if}
                        {#if task.cost_usd != null}<span>${task.cost_usd.toFixed(4)}</span>{/if}
                        {#if task.completed_at}
                          <span>at {formatTime(task.completed_at)}</span>
                        {/if}
                      </div>
                    {/if}

                    <!-- Always-visible reply bar -->
                    <div class="reply-bar">
                      <input
                        type="text"
                        class="reply-input"
                        placeholder={task.status === 'awaiting_input' ? 'Type your answer...' : task.status === 'running' ? 'Send correction...' : 'Send follow-up...'}
                        value={getReplyInput(task.task_id)}
                        on:input={(e) => setReplyInput(task.task_id, e.currentTarget.value)}
                        on:keydown={(e) => { if (e.key === 'Enter' && getReplyInput(task.task_id).trim()) handleReply(task.task_id); }}
                        disabled={replyingTask === task.task_id || task.status === 'pending'}
                      />
                      <button
                        class="btn btn-sm btn-primary"
                        on:click={() => handleReply(task.task_id)}
                        disabled={replyingTask === task.task_id || !getReplyInput(task.task_id).trim() || task.status === 'pending'}
                      >
                        {replyingTask === task.task_id ? '...' : 'Send'}
                      </button>
                      {#if task.status === 'running' || task.status === 'awaiting_input'}
                        <button
                          class="btn btn-sm btn-stop"
                          on:click|stopPropagation={() => handleStop(task.task_id)}
                          disabled={stoppingTask === task.task_id}
                          title="Stop this task"
                        >
                          {stoppingTask === task.task_id ? '...' : 'Stop'}
                        </button>
                      {/if}
                    </div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>

    {:else if activeTab === 'settings'}
      <!-- SETTINGS TAB -->
      <div class="tab-panel">
        {#if settingsLoading}
          <div class="loading">Loading settings...</div>
        {:else if settingsError}
          <div class="error-msg">{settingsError}</div>
        {:else if settings}
          <div class="settings-form">
            <div class="field">
              <label for="pet-name">Pet Name</label>
              <input type="text" id="pet-name" bind:value={settings.pet_name} />
            </div>

            <div class="field">
              <label for="pet-model">Pet Model (chat personality)</label>
              <select id="pet-model" bind:value={settings.pet_model}>
                {#each availableModels as model}
                  <option value={model}>{model}</option>
                {/each}
              </select>
            </div>

            <div class="field">
              <label for="worker-model">Worker Model (tasks)</label>
              <select id="worker-model" bind:value={settings.worker_model}>
                {#each availableModels as model}
                  <option value={model}>{model}</option>
                {/each}
              </select>
            </div>

            <div class="field">
              <label for="personality">Personality Prompt</label>
              <textarea
                id="personality"
                rows="4"
                bind:value={settings.personality_prompt}
                placeholder="Describe your pet's personality..."
              ></textarea>
            </div>

            <div class="field">
              <label for="max-turns">Max Conversation Turns</label>
              <input
                type="number"
                id="max-turns"
                bind:value={settings.max_conversation_turns}
                min="1"
                max="50"
              />
            </div>

            <div class="settings-actions">
              <button
                class="btn btn-primary"
                on:click={saveSettings}
                disabled={settingsSaving}
              >
                {settingsSaving ? 'Saving...' : 'Save Settings'}
              </button>
              {#if settingsSuccess}
                <span class="save-success">Saved!</span>
              {/if}
            </div>
          </div>
        {:else}
          <div class="loading">No settings loaded</div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .dashboard-root {
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

  .titlebar-title { font-size: 12px; color: var(--text-secondary); pointer-events: none; }
  .titlebar-controls { -webkit-app-region: no-drag; }
  .titlebar-btn { width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border: none; background: transparent; color: var(--text-secondary); border-radius: 4px; cursor: pointer; font-size: 14px; }
  .titlebar-btn.close:hover { background: var(--error); color: white; }

  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
    flex-shrink: 0;
    -webkit-app-region: no-drag;
  }

  .tab {
    flex: 1;
    padding: 8px 16px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 150ms ease;
  }

  .tab:hover { color: var(--text-primary); background: var(--bg-hover); }
  .tab.active { color: var(--accent-secondary); border-bottom-color: var(--accent-primary); }

  .tab-content {
    flex: 1;
    overflow-y: auto;
    -webkit-app-region: no-drag;
  }

  .tab-panel {
    padding: 16px;
  }

  /* State tab */
  .state-section { margin-bottom: 20px; }

  .state-section h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .mood-display {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  .mood-name { font-size: 16px; font-weight: 500; text-transform: capitalize; }
  .mood-values { font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); }

  .need-row, .emotion-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }

  .need-name, .emotion-name {
    width: 80px;
    font-size: 13px;
    text-transform: capitalize;
    color: var(--text-secondary);
  }

  .need-bar, .emotion-bar {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .need-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 300ms ease;
  }

  .need-fill.high { background: var(--success); }
  .need-fill.mid { background: var(--warning); }
  .need-fill.low { background: var(--error); }

  .emotion-fill {
    height: 100%;
    background: var(--accent-primary);
    border-radius: 4px;
    transition: width 300ms ease;
  }

  .need-value, .emotion-value {
    width: 40px;
    text-align: right;
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .behavior-name {
    font-size: 16px;
    text-transform: capitalize;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
    display: inline-block;
  }

  /* Actions tab */
  .actions-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .action-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
  }

  .action-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .action-name {
    font-weight: 500;
    text-transform: capitalize;
  }

  .action-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .action-stats {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .btn-sm {
    padding: 4px 10px;
    font-size: 12px;
  }

  /* Settings tab */
  .settings-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .field label {
    display: block;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .field input,
  .field select,
  .field textarea {
    width: 100%;
  }

  .settings-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .save-success {
    font-size: 13px;
    color: var(--success);
  }

  /* Worker tab */
  .worker-submit { margin-bottom: 16px; }

  .worker-input-row {
    display: flex;
    gap: 8px;
  }

  .worker-input {
    flex: 1;
  }

  .worker-tasks {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .worker-task-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .task-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    cursor: pointer;
    transition: background 150ms ease;
  }

  .task-header:hover { background: var(--bg-hover); }

  .task-meta {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .task-status {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .task-status.pending { background: rgba(234, 179, 8, 0.15); color: var(--warning); }
  .task-status.running { background: rgba(99, 102, 241, 0.15); color: var(--accent-primary); }
  .task-status.completed { background: rgba(34, 197, 94, 0.15); color: var(--success); }
  .task-status.failed { background: rgba(239, 68, 68, 0.15); color: var(--error); }
  .task-status.awaiting { background: rgba(251, 146, 60, 0.18); color: #fb923c; }

  .task-id {
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .task-time {
    font-size: 12px;
    color: var(--text-muted);
  }

  .task-details {
    padding: 0 12px 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .task-progress {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-top: 10px;
  }

  .progress-bar-track {
    flex: 1;
    height: 4px;
    background: var(--bg-tertiary);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar-fill {
    height: 100%;
    background: var(--accent-primary);
    border-radius: 2px;
  }

  .progress-bar-fill.indeterminate {
    width: 40%;
    animation: progress-slide 1.2s ease-in-out infinite;
  }

  @keyframes progress-slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
  }

  .progress-label {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
  }

  .task-result {
    padding-top: 10px;
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
    overflow-wrap: break-word;
  }

  /* Markdown body styles */
  .task-result :global(p) { margin: 0 0 8px; }
  .task-result :global(p:last-child) { margin-bottom: 0; }
  .task-result :global(h1),
  .task-result :global(h2),
  .task-result :global(h3),
  .task-result :global(h4) {
    margin: 12px 0 6px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .task-result :global(h1) { font-size: 16px; }
  .task-result :global(h2) { font-size: 15px; }
  .task-result :global(ul),
  .task-result :global(ol) {
    margin: 4px 0 8px;
    padding-left: 20px;
  }
  .task-result :global(li) { margin-bottom: 2px; }
  .task-result :global(code) {
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 1px 4px;
    background: var(--bg-tertiary);
    border-radius: 3px;
  }
  .task-result :global(pre) {
    margin: 6px 0;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.4;
  }
  .task-result :global(pre code) {
    padding: 0;
    background: none;
    border-radius: 0;
  }
  .task-result :global(blockquote) {
    margin: 6px 0;
    padding: 4px 10px;
    border-left: 3px solid var(--border-subtle);
    color: var(--text-secondary);
  }
  .task-result :global(a) {
    color: var(--accent-primary);
    text-decoration: underline;
  }
  .task-result :global(hr) {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 8px 0;
  }
  .task-result :global(strong) { font-weight: 600; }
  .task-result :global(table) {
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 12px;
  }
  .task-result :global(th),
  .task-result :global(td) {
    border: 1px solid var(--border-subtle);
    padding: 4px 8px;
    text-align: left;
  }
  .task-result :global(th) {
    background: var(--bg-tertiary);
    font-weight: 600;
  }

  .task-error {
    padding-top: 10px;
    font-size: 13px;
    color: var(--error);
  }

  .task-completed-time {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
  }

  .progress-feed {
    max-height: 300px;
    overflow-y: auto;
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .progress-entry {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    font-size: 12px;
    line-height: 1.4;
  }

  .progress-type-badge {
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(99, 102, 241, 0.12);
    color: var(--accent-primary);
  }

  .progress-type-badge.tool {
    background: rgba(34, 197, 94, 0.12);
    color: var(--success);
  }

  .progress-type-badge.thinking {
    background: rgba(234, 179, 8, 0.12);
    color: var(--warning);
  }

  .progress-type-badge.error {
    background: rgba(239, 68, 68, 0.12);
    color: var(--error);
  }

  .progress-content {
    color: var(--text-secondary);
    word-break: break-word;
  }

  .task-stats {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid var(--border-subtle);
  }

  /* Question cards (awaiting_input) */
  .question-card {
    margin-top: 10px;
    padding: 10px;
    background: rgba(251, 146, 60, 0.06);
    border: 1px solid rgba(251, 146, 60, 0.2);
    border-radius: 8px;
  }

  .question-text {
    font-size: 13px;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.4;
  }

  .question-options {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .btn-option {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 6px 10px;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    background: var(--bg-secondary);
    cursor: pointer;
    transition: all 150ms ease;
    text-align: left;
  }

  .btn-option:hover:not(:disabled):not(.selected) {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.08);
  }

  .btn-option.selected {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.15);
    box-shadow: 0 0 0 1px var(--accent-primary);
  }

  .btn-option:disabled { opacity: 0.5; cursor: not-allowed; }

  .send-answers-row {
    display: flex;
    justify-content: flex-end;
    margin-top: 8px;
  }

  .option-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .option-desc {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  /* Always-visible reply bar */
  .reply-bar {
    display: flex;
    gap: 6px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
  }

  .reply-input {
    flex: 1;
    font-size: 12px;
  }

  .btn-stop {
    background: transparent;
    border: 1px solid var(--error);
    color: var(--error);
    cursor: pointer;
    transition: all 150ms ease;
  }

  .btn-stop:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.12);
  }

  .btn-stop:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Clear completed row */
  .clear-row {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
    text-decoration: underline;
    padding: 2px 4px;
  }

  .btn-link:hover:not(:disabled) {
    color: var(--text-secondary);
  }

  .btn-link:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Task header right section */
  .task-header-right {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .btn-icon {
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: transparent;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    transition: all 150ms ease;
  }

  .btn-icon:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-star {
    color: var(--warning);
    font-size: 16px;
  }

  .btn-star:hover:not(:disabled) {
    background: rgba(234, 179, 8, 0.15);
  }

  .btn-delete {
    color: var(--text-muted);
    font-size: 12px;
  }

  .btn-delete:hover:not(:disabled) {
    color: var(--error);
    background: rgba(239, 68, 68, 0.1);
  }

  .starred-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(234, 179, 8, 0.15);
    color: var(--warning);
  }

  /* Common */
  .empty { color: var(--text-muted); font-size: 13px; }
  .loading { color: var(--text-muted); text-align: center; padding: 40px; }
  .error-msg {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--error);
    border-radius: 8px;
    padding: 8px 12px;
    color: var(--error);
    font-size: 13px;
    margin-bottom: 12px;
  }
</style>
