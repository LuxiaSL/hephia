<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { setupIPCListeners } from '$state/ipc-listener';
  import { systemContext } from '$state/stores';
  import {
    getActions,
    performAction,
    getSettings,
    updateSettings,
    getStateSnapshot,
    submitWorkerTask,
    getWorkerStatus,
  } from '$lib/tauri-commands';
  import type {
    SystemContext,
    SoulStatePayload,
    ActionInfo,
    PetSettings,
    WorkerTaskStatus,
  } from '$lib/types';

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
      const resp = await submitWorkerTask(task);
      workerInput = '';

      // Create initial status entry
      const status: WorkerTaskStatus = {
        task_id: resp.task_id,
        status: 'pending',
        result: null,
        error: null,
        created_at: Date.now() / 1000,
        completed_at: null,
      };
      workerTasks = [status, ...workerTasks];

      // Start polling for this task
      startPolling(resp.task_id);
    } catch (e) {
      workerError = `Failed to submit task: ${e}`;
    }
    workerSubmitting = false;
  }

  function startPolling(taskId: string) {
    const timer = window.setInterval(async () => {
      try {
        const status = await getWorkerStatus(taskId);
        workerTasks = workerTasks.map((t) =>
          t.task_id === taskId ? status : t,
        );

        // Stop polling when task is done
        if (status.status === 'completed' || status.status === 'failed') {
          stopPolling(taskId);
        }
      } catch {
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

  function toggleExpand(taskId: string) {
    expandedTask = expandedTask === taskId ? null : taskId;
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
  <div class="titlebar">
    <span class="titlebar-title">Hephia Dashboard</span>
    <div class="titlebar-controls">
      <button class="titlebar-btn close" on:click={handleClose}>✕</button>
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
                    >
                      {task.status}
                    </span>
                    <span class="task-id">{task.task_id.slice(0, 8)}</span>
                  </div>
                  <span class="task-time">{formatTime(task.created_at)}</span>
                </div>

                {#if expandedTask === task.task_id}
                  <div class="task-details">
                    {#if task.status === 'pending' || task.status === 'running'}
                      <div class="task-progress">
                        <div class="progress-bar-track">
                          <div class="progress-bar-fill" class:indeterminate={true}></div>
                        </div>
                        <span class="progress-label">
                          {task.status === 'pending' ? 'Queued...' : 'Working...'}
                        </span>
                      </div>
                    {:else if task.status === 'completed' && task.result}
                      <div class="task-result">{task.result}</div>
                    {:else if task.status === 'failed' && task.error}
                      <div class="task-error">{task.error}</div>
                    {/if}
                    {#if task.completed_at}
                      <div class="task-completed-time">
                        Completed at {formatTime(task.completed_at)}
                      </div>
                    {/if}
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
    white-space: pre-wrap;
    line-height: 1.5;
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
