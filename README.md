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

Each assignment is one Markdown file. `build.sh` turns it into both the web page and the PDF, so the two can't drift apart.

| Path | What it is |
| --- | --- |
| `content/<n>/index.md` | The write-up for assignment `<n>`, in Markdown with LaTeX math. |
| `content/<n>/media/` | That assignment's images. |
| `templates/page.html` | The Pandoc template every page is rendered through. |
| `static/style.css` | Styles shared by every page, including the `@media print` rules the PDF uses. |
| `build.sh` | Builds `content/` into `docs/` and `build/pdf/`. |
| `watch.sh` | Reruns `build.sh` whenever a source file is saved. |
| `docs/` | Generated site. GitHub Pages serves the `main` branch from this folder, so it is committed. |
| `build/pdf/` | Generated PDFs for Learn. Not committed. |
| `Assignment1.ipynb` | Code for Assignment 1: Newton-Raphson and golden section search for the distance from a point to a curve, least squares fitting for part 2, and the plots. |

Nothing in `docs/` is edited by hand. `build.sh` overwrites it.

## Building

Needs Pandoc and Chromium, both in the official repos:

```bash
sudo pacman -S pandoc-cli chromium
```

Then:

```bash
./build.sh        # every assignment
./build.sh 2      # just assignment 2
```

That writes `docs/<n>/index.html` for the site and `build/pdf/assignment-<n>.pdf` for Learn. The PDF is headless Chromium printing the same HTML the site serves, so the only differences between them are the `@media print` rules in `static/style.css`, which hide the navbar and force the light palette.

## Live preview

`watch.sh` rebuilds on every save, so the PDF you hand in is never stale. It needs `entr`, also in the official repos:

```bash
sudo pacman -S entr
```

```bash
./watch.sh        # watch every assignment
./watch.sh 1      # just assignment 1
```

It watches every `content/<n>/index.md` plus `build.sh`, `templates/page.html`, and `static/style.css`. Press space to force a rebuild, q or Ctrl-C to stop.

Open the PDF in a viewer that reloads when the file changes and it follows along a couple of seconds behind each save. Zathura does, and it needs a PDF backend:

```bash
sudo pacman -S zathura zathura-pdf-mupdf
zathura build/pdf/assignment-1.pdf &
```

## Previewing the site locally

Open it through a local server rather than straight from the file, so folder links like `1/` load their `index.html`:

```bash
python3 -m http.server 8000 --directory docs
```

Then go to http://localhost:8000.

## Running the notebook

The notebook needs Python 3 with NumPy, Matplotlib, and Jupyter.

```bash
pip install numpy matplotlib jupyterlab
jupyter lab Assignment1.ipynb
```

Running the plotting cells writes PNGs to `media/` in the repo root. That folder is ignored by git, so copy the images the page uses into the assignment's source folder:

```bash
cp media/*.png content/1/media/
```

## Adding an assignment

1. Create `content/<n>/index.md` with a `title` and `subtitle` in its YAML header, and put its images in `content/<n>/media/`.
2. Run `./build.sh`. The navbar is generated from the folders that exist, so every other assignment picks up the link automatically.
3. Commit `content/` and `docs/`, then push to `main`. Pages usually updates within a few minutes.
