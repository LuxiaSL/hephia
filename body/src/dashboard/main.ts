import { mount } from 'svelte';
import Dashboard from './Dashboard.svelte';

const app = mount(Dashboard, { target: document.getElementById('app')! });

export default app;
