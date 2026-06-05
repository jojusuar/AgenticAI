# REQUIREMENTS.md — Pomodoro Timer App

## Stack

- React 18 (Vite scaffold: `npm create vite@latest pomodoro -- --template react`)
- Plain CSS modules (no Tailwind, no UI libraries)
- Vitest + React Testing Library for tests

---

## Features

### 1. Timer display
- Shows `MM:SS` countdown (e.g. `25:00`)
- Counts down in real time once started
- Turns red when under 60 seconds remaining

### 2. Controls
- **Start / Pause** — single button that toggles
- **Reset** — restores current mode's full duration without switching mode

### 3. Modes
- **Focus** — 25 minutes
- **Short Break** — 5 minutes
- **Long Break** — 15 minutes
- Switching mode resets the timer automatically
- Active mode is visually indicated

### 4. Session counter
- Tracks how many Focus sessions have been completed (i.e. ran to 00:00)
- Displays as: `Sessions: 3`
- Resets to 0 when the page is refreshed (no persistence needed)

### 5. Page title sync
- `document.title` updates to reflect the current countdown and mode
- e.g. `[25:00] Focus — Pomodoro`

---

## File structure (expected output)

```
pomodoro/
├── index.html
├── vite.config.js
├── package.json
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── useTimer.js          # custom hook — all timer logic lives here
    ├── App.module.css
    └── __tests__/
        └── useTimer.test.js
```

---

## useTimer hook interface

The agent must implement a `useTimer` hook that exposes exactly this interface
(the tests will import it directly):

```js
const {
  secondsLeft,   // number
  isRunning,     // boolean
  mode,          // "focus" | "shortBreak" | "longBreak"
  sessions,      // number
  start,         // () => void
  pause,         // () => void
  reset,         // () => void
  setMode,       // (mode: string) => void
} = useTimer()
```

---

## Tests (useTimer.test.js)

Must include and pass the following test cases:

1. **Initial state** — starts at 25:00 (1500 seconds), not running, 0 sessions
2. **Mode switch** — calling `setMode("shortBreak")` sets `secondsLeft` to 300
3. **Countdown** — after calling `start()`, `secondsLeft` decreases by 1 per second
   (use `vi.useFakeTimers()`)
4. **Pause** — timer stops decrementing after `pause()` is called
5. **Reset** — `reset()` restores `secondsLeft` to the current mode's full duration
6. **Session count** — completing a full focus countdown increments `sessions` by 1

---

## Acceptance criteria

- `npm run dev` serves the app with no console errors
- `npm run test` passes all 6 tests
- The timer actually counts down visually in the browser
- Mode buttons highlight the active mode
- No TypeScript required — plain JS/JSX is fine

---

## What NOT to do

- Do not add a backend
- Do not use any state management library (no Redux, Zustand, etc.)
- Do not add sound, notifications, or animations
- Do not persist state to localStorage
- Keep total CSS under ~100 lines