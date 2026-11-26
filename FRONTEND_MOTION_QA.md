# Frontend motion QA checklist and verification

## Automated checks completed
- [x] TypeScript compiler API transpilation of `App.tsx` and `main.tsx` with JSX enabled (syntax diagnostics only; does not resolve imports or typecheck).
- [x] CSS parsed with `tinycss2`; no parse errors.
- [x] Scroll reveal falls back to visible content when reduced motion is enabled or `IntersectionObserver` is unavailable.
- [x] `prefers-reduced-motion` CSS overrides verified in source.

## Manual browser checks still required
- [ ] Verify initial intro overlay and first page reveal timing.
- [ ] Scroll through Command Center: hero, metrics, pipeline, panels and charts should reveal once without flashing.
- [ ] Switch every navigation page; new content should reveal after navigation.
- [ ] Switch NSL-KDD / UNSW-NB15; refreshed page sections should reveal.
- [ ] Confirm the orange scroll progress line tracks document scroll.
- [ ] Confirm hover motion does not shift layout or obscure controls.
- [ ] Test 1440px, 1024px, 768px and 390px widths.
- [ ] Enable `prefers-reduced-motion: reduce`; content should be visible without animated entrance.
- [ ] Confirm all buttons, forms, tables, charts and backend workflows remain functional.

## Build limitation
`npm install --no-audit --no-fund` timed out in this environment. Consequently a full TypeScript typecheck and Vite production build could not be completed, and no real-browser visual QA is claimed.
