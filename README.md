# SYDE 572

My assignments for SYDE 572 at the University of Waterloo, Fall 2026. The write-ups are published as a website at [lukeypookster.com/SYDE572](https://lukeypookster.com/SYDE572/).

## Assignments

| # | Topic | Page |
| --- | --- | --- |
| 1 | Shortest distance from a point to a function, and fitting to an equation | [Assignment 1](https://lukeypookster.com/SYDE572/1/) |
| 2 | Not posted yet | |
| 3 | Not posted yet | |
| 4 | Not posted yet | |
| 5 | Not posted yet | |

## Layout

| Path | What it is |
| --- | --- |
| `Assignment1.ipynb` | Code for Assignment 1: Newton-Raphson and golden section search for the distance from a point to a curve, plus the plots of each method's intermediate steps. |
| `docs/` | The website. GitHub Pages serves the `main` branch from this folder. |
| `docs/index.html` | Redirects to Assignment 1 until there is more than one assignment. |
| `docs/1/` | The Assignment 1 page and its images in `docs/1/media/`. |
| `docs/style.css` | Styles shared by every page. |
| `docs/site.js` | The tabs and the click-to-enlarge image viewer. |

## Running the notebook

The notebook needs Python 3 with NumPy, Matplotlib, and Jupyter.

```bash
pip install numpy matplotlib jupyterlab
jupyter lab Assignment1.ipynb
```

Running the plotting cells writes PNGs to `media/` in the repo root. That folder is ignored by git, so copy the images the page uses into the site:

```bash
cp media/*.png docs/1/media/
```

## Previewing the site locally

Open it through a local server rather than straight from the file, so folder links like `1/` load their `index.html`:

```bash
python3 -m http.server 8000 --directory docs
```

Then go to http://localhost:8000.

## Adding an assignment

1. Create `docs/<n>/index.html`, using `docs/1/index.html` as a starting point, and put its images in `docs/<n>/media/`.
2. The navbar is copied into each page, so in every page change that assignment's `<span aria-disabled="true">` into a link.
3. Push to `main`. Pages usually updates within a few minutes.
