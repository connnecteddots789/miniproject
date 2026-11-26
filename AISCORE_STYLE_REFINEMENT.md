# SENTINEL — AIScore-inspired motion and layout refinement

## Direction
The AIScore reference was used as inspiration for its editorial pacing, spacious section hierarchy, oversized typography, restrained visual language, and scroll-led reveals. SENTINEL retains its graphite/black surfaces, orange accent system, brand mark, navigation, research content, APIs, datasets, and existing application workflows.

## Implemented in this pass
- Added a thin orange page-scroll progress indicator.
- Added IntersectionObserver-based section reveals with a staggered rhythm for metrics and content blocks.
- Added restrained blur-to-sharp and upward entrance transitions for page headings, hero sections, metric groups, panels, charts, and explanatory sections.
- Added subtle hover lift and border emphasis to interactive surfaces.
- Refined the Command Center into a more editorial hero/metrics rhythm with larger typography, wider section spacing, flatter borders, and less repetitive card treatment.
- Refined navigation and controls to use the existing orange/graphite design tokens.
- Added reduced-motion handling so visitors who request reduced motion do not receive scroll animations.
- Kept the motion layer dependency-free; no new runtime package or backend dependency was introduced.

## Explicitly preserved
- All existing routes/pages and navigation items.
- FastAPI endpoints, model training, prediction, dataset handling, simulation, experiments, benchmark and research validation workflows.
- Existing black/graphite and orange identity, SENTINEL logo, and product naming.
- Existing API polling and persisted artifacts.

## Verification
- Backend regression suite: `23 passed, 4 warnings`.
- Frontend production build: **not verified**. The extracted project does not include `node_modules`; package installation could not complete in the offline environment, so Vite/TypeScript could not be run with the React dependencies present.
- Real-browser scroll/animation QA: **not performed**. The effects are implemented in CSS and IntersectionObserver, but need to be viewed in the user's browser after dependencies are installed.

## Run locally
From `frontend/`, run `npm install` and `npm run build`, then start the frontend with `npm run dev`. Start the backend using the existing project instructions. Review the scroll reveals at desktop and mobile widths, and with the operating system's reduced-motion preference enabled.
