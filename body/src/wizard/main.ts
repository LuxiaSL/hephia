import { mount } from 'svelte';
import Wizard from './Wizard.svelte';

const app = mount(Wizard, { target: document.getElementById('app')! });

export default app;
