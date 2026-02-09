<script lang="ts">
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import {
    checkEnvironment,
    startBackend,
    updateSettings,
    markWizardComplete,
  } from '$lib/tauri-commands';
  import type { EnvironmentStatus, PetSettings } from '$lib/types';

  let step = 0;
  let envStatus: EnvironmentStatus | null = null;
  let checking = false;
  let launching = false;
  let error = '';

  // Form data
  let anthropicKey = '';
  let openaiKey = '';
  let openrouterKey = '';
  let petModel = 'haiku-4.5';
  let workerModel = 'opus-4.6';
  let petName = 'Hephia';
  let personalityPrompt = '';

  const steps = ['Welcome', 'Environment', 'API Keys', 'Models', 'Identity', 'Launch'];

  // Computed: can proceed to next step
  function canProceed(): boolean {
    if (step === 1) {
      // Need all env checks passing
      return !!envStatus && envStatus.python_found && envStatus.uv_found;
    }
    if (step === 2) {
      // Need at least one API key
      return !!(anthropicKey.trim() || openaiKey.trim() || openrouterKey.trim());
    }
    if (step === 4) {
      // Need a pet name
      return !!petName.trim();
    }
    return true;
  }

  async function checkEnv() {
    checking = true;
    error = '';
    try {
      envStatus = await checkEnvironment();
    } catch (e) {
      error = `Environment check failed: ${e}`;
    }
    checking = false;
  }

  async function handleLaunch() {
    launching = true;
    error = '';
    try {
      // TODO: Write API keys to backend .env file
      // This needs a Rust command to write to the soul server's config
      // For now we assume keys are set in the environment

      // Start the backend (treat "already running" as success)
      try {
        await startBackend();
      } catch (backendErr) {
        if (!String(backendErr).includes('already running')) {
          throw backendErr;
        }
      }

      // Apply settings via the API (backend is now running)
      try {
        const settings: PetSettings = {
          pet_name: petName,
          pet_model: petModel,
          worker_model: workerModel,
          personality_prompt: personalityPrompt,
          memory_significance_threshold: 0.5,
          max_conversation_turns: 20,
        };
        await updateSettings(settings);
      } catch (settingsErr) {
        // Non-fatal: settings can be configured later from dashboard
        console.warn('Failed to apply wizard settings:', settingsErr);
      }

      // Mark wizard as completed
      await markWizardComplete();

      // Close wizard — the main app will detect backend is running
      const win = getCurrentWindow();
      await win.close();
    } catch (e) {
      error = `Launch failed: ${e}`;
      launching = false;
    }
  }

  function nextStep() {
    if (step < steps.length - 1 && canProceed()) step++;
  }

  function prevStep() {
    if (step > 0) step--;
  }
</script>

<div class="wizard-root">
  <!-- Progress indicator -->
  <div class="progress-bar">
    {#each steps as s, i}
      <div class="progress-step" class:active={i === step} class:done={i < step}>
        <div class="step-dot">{i < step ? '\u2713' : i + 1}</div>
        <span class="step-label">{s}</span>
      </div>
      {#if i < steps.length - 1}
        <div class="step-line" class:active={i < step}></div>
      {/if}
    {/each}
  </div>

  <!-- Step content -->
  <div class="step-content">
    {#if step === 0}
      <!-- Welcome -->
      <div class="welcome">
        <h1>Welcome to Hephia</h1>
        <p>Hephia is a desktop companion that remembers, feels, and thinks.</p>
        <p>Let's get everything set up.</p>
      </div>

    {:else if step === 1}
      <!-- Environment -->
      <div class="environment">
        <h2>Environment Check</h2>
        <p class="hint">Hephia needs Python and uv to run its backend.</p>
        {#if !envStatus && !checking}
          <button class="btn btn-primary" on:click={checkEnv}>Check Environment</button>
        {:else if checking}
          <p>Checking...</p>
        {:else if envStatus}
          <div class="env-list">
            <div class="env-item" class:found={envStatus.python_found} class:missing={!envStatus.python_found}>
              <span>Python</span>
              <span>{envStatus.python_found ? envStatus.python_version : 'Not found'}</span>
            </div>
            <div class="env-item" class:found={envStatus.uv_found} class:missing={!envStatus.uv_found}>
              <span>uv</span>
              <span>{envStatus.uv_found ? envStatus.uv_version : 'Not found'}</span>
            </div>
            <div class="env-item" class:found={envStatus.venv_exists} class:missing={!envStatus.venv_exists}>
              <span>Virtual Environment</span>
              <span>{envStatus.venv_exists ? 'Found' : 'Will be created'}</span>
            </div>
            <div class="env-item" class:found={envStatus.deps_installed} class:missing={!envStatus.deps_installed}>
              <span>Dependencies</span>
              <span>{envStatus.deps_installed ? 'Installed' : 'Will be installed'}</span>
            </div>
          </div>
          {#if !envStatus.python_found || !envStatus.uv_found}
            <p class="error-hint">
              {#if !envStatus.python_found}Python{/if}
              {#if !envStatus.python_found && !envStatus.uv_found} and {/if}
              {#if !envStatus.uv_found}uv{/if}
              must be installed to continue.
            </p>
          {/if}
        {/if}
      </div>

    {:else if step === 2}
      <!-- API Keys -->
      <div class="api-keys">
        <h2>API Keys</h2>
        <p class="hint">Configure your LLM provider API keys. You need at least one.</p>
        <div class="field">
          <label for="anthropic-key">Anthropic API Key</label>
          <input type="password" id="anthropic-key" bind:value={anthropicKey} placeholder="sk-ant-..." />
        </div>
        <div class="field">
          <label for="openai-key">OpenAI API Key (optional)</label>
          <input type="password" id="openai-key" bind:value={openaiKey} placeholder="sk-..." />
        </div>
        <div class="field">
          <label for="openrouter-key">OpenRouter API Key (optional)</label>
          <input type="password" id="openrouter-key" bind:value={openrouterKey} placeholder="sk-or-..." />
        </div>
        {#if !canProceed()}
          <p class="error-hint">At least one API key is required.</p>
        {/if}
      </div>

    {:else if step === 3}
      <!-- Models -->
      <div class="models">
        <h2>Model Selection</h2>
        <p class="hint">Choose which models Hephia uses. You can change these later in Settings.</p>
        <div class="field">
          <label for="pet-model">Pet Model (chat personality)</label>
          <select id="pet-model" bind:value={petModel}>
            <option value="haiku-4.5">Claude Haiku 4.5 (Recommended)</option>
            <option value="haiku">Claude 3.5 Haiku</option>
          </select>
        </div>
        <div class="field">
          <label for="worker-model">Worker Model (tasks)</label>
          <select id="worker-model" bind:value={workerModel}>
            <option value="opus-4.6">Claude Opus 4.6 (Recommended)</option>
            <option value="sonnet-4.5">Claude Sonnet 4.5</option>
          </select>
        </div>
      </div>

    {:else if step === 4}
      <!-- Identity -->
      <div class="identity">
        <h2>Pet Identity</h2>
        <div class="field">
          <label for="pet-name-input">Name</label>
          <input type="text" id="pet-name-input" bind:value={petName} />
        </div>
        <div class="field">
          <label for="personality-input">Personality Prompt (optional)</label>
          <textarea id="personality-input" rows="4" bind:value={personalityPrompt} placeholder="Describe your pet's personality..."></textarea>
        </div>
      </div>

    {:else if step === 5}
      <!-- Launch -->
      <div class="launch">
        <h2>Ready to Launch</h2>
        <div class="summary">
          <div class="summary-row">
            <span class="summary-label">Name</span>
            <span class="summary-value">{petName}</span>
          </div>
          <div class="summary-row">
            <span class="summary-label">Pet Model</span>
            <span class="summary-value">{petModel}</span>
          </div>
          <div class="summary-row">
            <span class="summary-label">Worker Model</span>
            <span class="summary-value">{workerModel}</span>
          </div>
        </div>
        <p>Everything is configured. Click below to start {petName}.</p>
        {#if error}
          <div class="error-msg">{error}</div>
        {/if}
        <button class="btn btn-primary launch-btn" on:click={handleLaunch} disabled={launching}>
          {launching ? 'Starting...' : `Launch ${petName}`}
        </button>
      </div>
    {/if}
  </div>

  <!-- Navigation -->
  <div class="nav-bar">
    <button class="btn" on:click={prevStep} disabled={step === 0 || launching}>Back</button>
    <span class="step-indicator">{step + 1} / {steps.length}</span>
    {#if step < steps.length - 1}
      <button class="btn btn-primary" on:click={nextStep} disabled={!canProceed()}>Next</button>
    {/if}
  </div>
</div>

<style>
  .wizard-root {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    padding: 24px;
  }

  .progress-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 32px;
    flex-shrink: 0;
  }

  .progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: var(--text-muted);
    transition: all 200ms ease;
  }

  .progress-step.active .step-dot {
    border-color: var(--accent-primary);
    background: var(--accent-primary);
    color: white;
  }

  .progress-step.done .step-dot {
    border-color: var(--success);
    background: var(--success);
    color: white;
  }

  .step-label { font-size: 10px; color: var(--text-muted); }

  .step-line {
    width: 40px;
    height: 2px;
    background: var(--border-subtle);
    margin: 0 4px;
    margin-bottom: 18px;
  }

  .step-line.active { background: var(--success); }

  .step-content { flex: 1; overflow-y: auto; }
  .step-content h1 { font-size: 24px; margin-bottom: 12px; }
  .step-content h2 { font-size: 18px; margin-bottom: 16px; }
  .step-content p { color: var(--text-secondary); margin-bottom: 12px; }

  .field { margin-bottom: 16px; }

  .field label {
    display: block;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .hint { font-size: 13px; color: var(--text-muted); margin-bottom: 16px; }
  .error-hint { font-size: 13px; color: var(--error); margin-top: 12px; }

  .env-list { display: flex; flex-direction: column; gap: 8px; }

  .env-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
    border-left: 3px solid var(--border-subtle);
  }

  .env-item.found { border-left-color: var(--success); }
  .env-item.missing { border-left-color: var(--error); }

  .error-msg {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--error);
    border-radius: 8px;
    padding: 8px 12px;
    color: var(--error);
    font-size: 13px;
    margin-bottom: 12px;
  }

  .welcome { text-align: center; padding-top: 40px; }
  .launch { text-align: center; padding-top: 20px; }

  .summary {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    text-align: left;
    display: inline-flex;
    flex-direction: column;
    gap: 6px;
  }

  .summary-row {
    display: flex;
    gap: 12px;
  }

  .summary-label {
    font-size: 13px;
    color: var(--text-muted);
    width: 100px;
  }

  .summary-value {
    font-size: 13px;
    color: var(--text-primary);
  }

  .launch-btn {
    margin-top: 8px;
    padding: 10px 32px;
    font-size: 15px;
  }

  .nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
    flex-shrink: 0;
  }

  .step-indicator { font-size: 12px; color: var(--text-muted); }
</style>
