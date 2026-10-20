"""A lightweight, persistent desktop labeler for folders of bird images.

Run directly from the project environment:

    python scripts/image_labeler_gui.py

Labels and user-added classes are stored immediately in ``.bird_labeler.sqlite3``
inside the selected image folder.  The image files themselves are never changed.
"""

from __future__ import annotations

import csv
import random
import re
import sqlite3
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DATABASE_FILENAME = ".bird_labeler.sqlite3"
DEFAULT_CLASSES = (
    "Blaumeise",
    "Eichelhäher",
    "Feldsperling",
    "Kleiber",
    "Kohlmeise",
    "No Bird",
    "Rotkehlchen",
    "Specht",
    "Sumpfmeise",
    "Tannenmeise",
    "Taube",
)
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
PREVIEW_SIZE = (1200, 760)


def date_from_filename(filename: str) -> str:
    """Extract a YYYY-MM-DD date when a filename contains a valid date."""
    match = DATE_PATTERN.search(filename)
    if match is None:
        return ""
    try:
        return datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=UTC
        ).date().isoformat()
    except ValueError:
        return ""


class LabelStore:
    """Transactional, folder-local persistence for labels and classes."""

    def __init__(self, image_folder: Path) -> None:
        self.path = image_folder / DATABASE_FILENAME
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS classes (
                name TEXT PRIMARY KEY COLLATE NOCASE,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS labels (
                relative_path TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(label) REFERENCES classes(name)
            )
            """
        )
        with self.connection:
            for class_name in DEFAULT_CLASSES:
                self.connection.execute(
                    "INSERT OR IGNORE INTO classes(name, created_at) VALUES (?, ?)",
                    (class_name, self.now()),
                )

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def close(self) -> None:
        self.connection.close()

    def classes(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute("SELECT name FROM classes ORDER BY name COLLATE NOCASE")
        ]

    def add_class(self, class_name: str) -> bool:
        cleaned_name = class_name.strip()
        if not cleaned_name:
            raise ValueError("A class name cannot be empty.")
        with self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO classes(name, created_at) VALUES (?, ?)",
                (cleaned_name, self.now()),
            )
        return cursor.rowcount == 1

    def labels(self) -> dict[str, str]:
        return dict(self.connection.execute("SELECT relative_path, label FROM labels"))

    def set_label(self, relative_path: str, label: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO labels(relative_path, label, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    label = excluded.label,
                    updated_at = excluded.updated_at
                """,
                (relative_path, label, self.now()),
            )

    def clear_label(self, relative_path: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM labels WHERE relative_path = ?", (relative_path,))


class ImageLabelerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bird Image Labeler")
        self.root.minsize(850, 650)
        self.root.geometry("1200x900")

        self.folder: Path | None = None
        self.store: LabelStore | None = None
        self.image_paths: list[Path] = []
        self.labels: dict[str, str] = {}
        self.current_index = 0
        self.preview_image: ImageTk.PhotoImage | None = None
        self.class_choice = tk.StringVar()
        self.status_value = tk.StringVar(value="Choose a folder of images to begin.")
        self.image_info_value = tk.StringVar(value="No folder selected")
        self.label_value = tk.StringVar(value="Current label: not labelled")
        self.shuffle_on_load = tk.BooleanVar(value=True)
        self._build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(toolbar, text="Choose image folder…", command=self.choose_folder).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Shuffle", command=self.shuffle_images).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(toolbar, text="Shuffle on load", variable=self.shuffle_on_load).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Button(toolbar, text="Export CSV…", command=self.export_csv).pack(side=tk.RIGHT)

        class_bar = ttk.Frame(container)
        class_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(class_bar, text="Apply class:").pack(side=tk.LEFT)
        self.class_selector = ttk.Combobox(
            class_bar, textvariable=self.class_choice, state="readonly", width=30
        )
        self.class_selector.pack(side=tk.LEFT, padx=(8, 8))
        self.class_selector.bind("<<ComboboxSelected>>", lambda _event: self.apply_selected_class())
        self.apply_button = ttk.Button(
            class_bar, text="Save label", command=self.apply_selected_class, state=tk.DISABLED
        )
        self.apply_button.pack(side=tk.LEFT)
        ttk.Button(class_bar, text="Add class…", command=self.add_class).pack(side=tk.LEFT, padx=(8, 0))
        self.clear_button = ttk.Button(
            class_bar, text="Clear label", command=self.clear_current_label, state=tk.DISABLED
        )
        self.clear_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(container, textvariable=self.image_info_value).pack(anchor=tk.W)
        ttk.Label(container, textvariable=self.label_value).pack(anchor=tk.W, pady=(2, 8))
        self.preview = ttk.Label(
            container, text="Choose an image folder to begin", anchor=tk.CENTER, relief=tk.SUNKEN
        )
        self.preview.pack(fill=tk.BOTH, expand=True)

        navigation = ttk.Frame(container)
        navigation.pack(fill=tk.X, pady=(8, 0))
        self.previous_button = ttk.Button(
            navigation, text="← Previous", command=self.previous_image, state=tk.DISABLED
        )
        self.previous_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(
            navigation, text="Next →", command=self.next_image, state=tk.DISABLED
        )
        self.next_button.pack(side=tk.LEFT, padx=(8, 0))
        self.next_unlabelled_button = ttk.Button(
            navigation,
            text="Next unlabelled",
            command=self.next_unlabelled_image,
            state=tk.DISABLED,
        )
        self.next_unlabelled_button.pack(side=tk.LEFT, padx=(18, 0))
        ttk.Label(navigation, textvariable=self.status_value).pack(side=tk.RIGHT)

        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<space>", lambda _event: self.next_image())
        self.root.bind("<n>", lambda _event: self.next_unlabelled_image())

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="Choose a folder containing images")
        if selected:
            self.load_folder(Path(selected))

    def load_folder(self, folder: Path) -> None:
        image_paths = sorted(
            path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            messagebox.showinfo("Bird Image Labeler", "No supported images were found in this folder.")
            return
        if self.store is not None:
            self.store.close()
        self.folder = folder
        self.store = LabelStore(folder)
        self.image_paths = image_paths
        self.labels = self.store.labels()
        if self.shuffle_on_load.get():
            random.shuffle(self.image_paths)
        self.current_index = 0
        self.refresh_class_choices()
        self.enable_controls()
        self.show_current_image()

    def refresh_class_choices(self) -> None:
        if self.store is None:
            return
        classes = self.store.classes()
        self.class_selector["values"] = classes
        if self.class_choice.get() not in classes:
            self.class_choice.set(classes[0] if classes else "")

    def enable_controls(self) -> None:
        for widget in (
            self.apply_button,
            self.clear_button,
            self.previous_button,
            self.next_button,
            self.next_unlabelled_button,
        ):
            widget.configure(state=tk.NORMAL)

    def relative_path(self, path: Path) -> str:
        assert self.folder is not None
        return path.relative_to(self.folder).as_posix()

    def show_current_image(self) -> None:
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        relative_path = self.relative_path(path)
        try:
            with Image.open(path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        except (OSError, UnidentifiedImageError) as error:
            self.preview_image = None
            self.preview.configure(image="", text=f"Could not open {path.name}: {error}")
        else:
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_image, text="")
        current_label = self.labels.get(relative_path)
        self.image_info_value.set(
            f"{self.current_index + 1:,} / {len(self.image_paths):,}  —  {relative_path}"
        )
        self.label_value.set(
            f"Current label: {current_label}" if current_label else "Current label: not labelled"
        )
        labelled_count = sum(1 for image_path in self.image_paths if self.relative_path(image_path) in self.labels)
        self.status_value.set(f"{labelled_count:,} / {len(self.image_paths):,} labelled")
        if current_label:
            self.class_choice.set(current_label)

    def apply_selected_class(self) -> None:
        if self.store is None or not self.image_paths:
            return
        class_name = self.class_choice.get()
        if not class_name:
            messagebox.showwarning("Bird Image Labeler", "Add or select a class first.")
            return
        relative_path = self.relative_path(self.image_paths[self.current_index])
        self.store.set_label(relative_path, class_name)
        self.labels[relative_path] = class_name
        self.show_current_image()
        self.next_image()

    def clear_current_label(self) -> None:
        if self.store is None or not self.image_paths:
            return
        relative_path = self.relative_path(self.image_paths[self.current_index])
        self.store.clear_label(relative_path)
        self.labels.pop(relative_path, None)
        self.show_current_image()

    def add_class(self) -> None:
        if self.store is None:
            messagebox.showwarning("Bird Image Labeler", "Choose an image folder first.")
            return
        class_name = simpledialog.askstring("Add class", "Class name:", parent=self.root)
        if class_name is None:
            return
        try:
            added = self.store.add_class(class_name)
        except ValueError as error:
            messagebox.showwarning("Bird Image Labeler", str(error))
            return
        self.refresh_class_choices()
        self.class_choice.set(class_name.strip())
        self.status_value.set(
            f"Added class: {class_name.strip()}" if added else "That class already exists."
        )

    def previous_image(self) -> None:
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_current_image()

    def next_image(self) -> None:
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_current_image()

    def next_unlabelled_image(self) -> None:
        if not self.image_paths:
            return
        for offset in range(1, len(self.image_paths) + 1):
            index = (self.current_index + offset) % len(self.image_paths)
            if self.relative_path(self.image_paths[index]) not in self.labels:
                self.current_index = index
                self.show_current_image()
                return
        messagebox.showinfo("Bird Image Labeler", "Every image in this folder is labelled.")

    def shuffle_images(self) -> None:
        if not self.image_paths:
            return
        current_path = self.image_paths[self.current_index]
        random.shuffle(self.image_paths)
        self.current_index = self.image_paths.index(current_path)
        self.show_current_image()

    def export_csv(self) -> None:
        if self.folder is None:
            messagebox.showwarning("Bird Image Labeler", "Choose an image folder first.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export labels as CSV",
            initialdir=self.folder,
            initialfile="bird_labels.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not destination:
            return
        rows = [
            (self.relative_path(path), date_from_filename(path.name), self.labels[self.relative_path(path)])
            for path in self.image_paths
            if self.relative_path(path) in self.labels
        ]
        with Path(destination).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(("filename", "date", "label"))
            writer.writerows(sorted(rows, key=lambda row: row[0].lower()))
        self.status_value.set(f"Exported {len(rows):,} labels to {Path(destination).name}")

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ImageLabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
