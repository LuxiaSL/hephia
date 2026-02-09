/**
 * Pet overlay window entry point.
 *
 * Initializes the WebGL creature renderer, listens for IPC state events,
 * and manages thought bubbles and the context menu.
 */

import { mount } from 'svelte';
import Overlay from './Overlay.svelte';

const app = mount(Overlay, { target: document.getElementById('app')! });

export default app;
