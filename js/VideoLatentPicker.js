import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "VideoLatentPicker";
const NUMBER_TOKEN = /\[0-9\]\+?/;
const pickers = new Set();

let lastQueueRemaining = null;
api.addEventListener("status", ({ detail }) => {
    const remaining = detail?.exec_info?.queue_remaining;
    if (remaining === 0 && lastQueueRemaining !== 0) {
        pickers.forEach((p) => p.refresh());
    }
    lastQueueRemaining = remaining ?? lastQueueRemaining;
});

function el(tag, style = {}, text) {
    const e = document.createElement(tag);
    Object.assign(e.style, style);
    if (text !== undefined) e.textContent = text;
    return e;
}

function button(label, onClick, style = {}) {
    const b = el("button", Object.assign({ fontSize: "11px", padding: "1px 6px", cursor: "pointer", whiteSpace: "nowrap" }, style), label);
    b.addEventListener("click", (e) => {
        e.stopPropagation();
        onClick();
    });
    return b;
}

function videoUrl(v) {
    const q = new URLSearchParams({ filename: v.filename, subfolder: v.subfolder, type: "output" });
    return api.apiURL(`/view?${q}`);
}

class VideoPopup {
    static instance = null;

    static get() {
        if (!VideoPopup.instance) {
            VideoPopup.instance = new VideoPopup();
        }
        return VideoPopup.instance;
    }

    constructor() {
        this.currentVideo = null;
        this.currentPicker = null;

        this.overlay = el("div", {
            display: "none",
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            zIndex: "10000",
            justifyContent: "center",
            alignItems: "center",
            boxSizing: "border-box",
            padding: "20px",
        });

        this.dialog = el("div", {
            backgroundColor: "var(--comfy-menu-bg, #242424)",
            border: "1px solid var(--border-color, #444)",
            borderRadius: "8px",
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.6)",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            maxWidth: "90vw",
            maxHeight: "90vh",
            padding: "10px 14px 14px 14px",
            color: "var(--input-text, #ddd)",
            boxSizing: "border-box",
        });

        const header = el("div", {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            justifyContent: "space-between",
            width: "100%",
        });

        this.title = el("span", {
            fontWeight: "bold",
            fontSize: "12px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: "1",
        });

        const actions = el("div", {
            display: "flex",
            alignItems: "center",
            gap: "6px",
            flexShrink: "0",
        });

        this.selectBtn = button("Select latent", () => {
            if (this.currentVideo && this.currentPicker) {
                this.currentPicker.select(this.currentVideo);
                this.selectBtn.textContent = "Selected";
            }
        });

        this.deleteBtn = button("Delete", () => {
            if (this.currentVideo && this.currentPicker) {
                this.currentPicker.deleteVideo(this.currentVideo);
            }
        }, { color: "#ff6b6b" });

        const closeBtn = button("✕", () => this.close(), {
            fontSize: "14px",
            fontWeight: "bold",
            padding: "1px 6px",
        });
        closeBtn.title = "Close preview (Esc)";

        actions.append(this.selectBtn, this.deleteBtn, closeBtn);
        header.append(this.title, actions);

        this.video = el("video", {
            maxWidth: "85vw",
            maxHeight: "75vh",
            backgroundColor: "#000",
            borderRadius: "4px",
            display: "block",
        });
        this.video.controls = true;

        this.dialog.append(header, this.video);
        this.overlay.append(this.dialog);

        // Prevent ComfyUI canvas / node interaction while overlay is open
        const stop = (e) => e.stopPropagation();
        ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel", "contextmenu"].forEach((evt) => {
            this.overlay.addEventListener(evt, stop);
        });

        // Close on clicking backdrop
        this.overlay.addEventListener("click", (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        });

        // Close on Escape key
        window.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && this.isOpen()) {
                e.stopPropagation();
                this.close();
            }
        }, true);

        document.body.appendChild(this.overlay);
    }

    isOpen() {
        return this.overlay.style.display === "flex";
    }

    open(v, picker) {
        this.currentVideo = v;
        this.currentPicker = picker;
        this.title.textContent = v.path;
        this.title.title = v.path;
        const isSel = picker?.widget("selected_video")?.value === v.path;
        this.selectBtn.textContent = isSel ? "Unselect" : "Select";
        this.video.src = videoUrl(v);
        this.overlay.style.display = "flex";
        this.video.play().catch(() => {});
    }

    close() {
        if (!this.isOpen()) return;
        this.video.pause();
        this.video.removeAttribute("src");
        this.video.load();
        this.overlay.style.display = "none";
        this.currentVideo = null;
        this.currentPicker = null;
    }
}

class Picker {
    constructor(node) {
        this.node = node;
        this.videos = [];
        this.showThumbs = false;
        this.widget = (name) => node.widgets?.find((w) => w.name === name);

        this.root = el("div", {
            display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%",
            boxSizing: "border-box", fontSize: "11px", color: "var(--input-text, #ddd)", overflow: "hidden",
        });

        const toolbar = el("div", { display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" });
        this.thumbBtn = button("Show thumbnails", () => this.toggleThumbs());
        this.status = el("span", { opacity: "0.7" });
        toolbar.append(button("Refresh", () => this.refresh()), this.thumbBtn, this.status);

        this.latentLabel = el("div", { opacity: "0.85", wordBreak: "break-all" });
        this.list = el("div", { flex: "1", overflowY: "auto", display: "flex", flexDirection: "column", gap: "2px", minHeight: "0" });

        this.root.append(toolbar, this.latentLabel, this.list);
        // Keep canvas from hijacking wheel/drag inside the list
        this.list.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
    }

    async refresh() {
        const regex = this.widget("video_regex")?.value ?? "";
        this.status.textContent = "Loading...";
        try {
            const res = await api.fetchApi(`/martinodes/videos?${new URLSearchParams({ regex })}`);
            const data = await res.json();
            this.videos = data.videos ?? [];
            this.status.textContent = data.error ?? `${this.videos.length} video(s)`;
        } catch (e) {
            this.videos = [];
            this.status.textContent = `Error: ${e.message}`;
        }
        this.render();
    }

    toggleThumbs() {
        this.showThumbs = !this.showThumbs;
        this.thumbBtn.textContent = this.showThumbs ? "Hide thumbnails" : "Show thumbnails";
        this.render();
    }

    play(v) {
        VideoPopup.get().open(v, this);
    }

    closePlayer() {
        if (VideoPopup.instance?.currentPicker === this) {
            VideoPopup.instance.close();
        }
    }

    select(v) {
        const w = this.widget("selected_video");
        if (!w) return;
        w.value = v.path;
        w.callback?.(w.value);
        this.render();
        this.node.setDirtyCanvas(true, true);
    }

    async deleteVideo(v) {
        const latentPath = this.latentFor(v.number);
        const confirmMsg = `Are you sure you want to delete this video and its latent file?\n\nVideo: ${v.path}\nLatent: ${latentPath || "(none)"}`;
        if (!confirm(confirmMsg)) return;

        if (VideoPopup.instance?.currentVideo?.path === v.path) {
            VideoPopup.instance.close();
        }

        this.status.textContent = `Deleting ${v.filename}...`;

        try {
            const res = await api.fetchApi("/martinodes/delete_video_and_latent", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_path: v.path,
                    video_regex: this.widget("video_regex")?.value ?? "",
                    latent_pattern: this.widget("latent_pattern")?.value ?? "",
                    latent_path: latentPath,
                }),
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                alert(`Error deleting: ${data.error || res.statusText}`);
                this.status.textContent = `Delete failed: ${data.error || res.statusText}`;
                return;
            }

            const selWidget = this.widget("selected_video");
            if (selWidget && selWidget.value === v.path) {
                selWidget.value = "";
                selWidget.callback?.("");
                this.node.setDirtyCanvas(true, true);
            }

            pickers.forEach((p) => p.refresh());
        } catch (e) {
            alert(`Error deleting: ${e.message}`);
            this.status.textContent = `Delete failed: ${e.message}`;
        }
    }

    latentFor(number) {
        const pattern = this.widget("latent_pattern")?.value ?? "";
        return pattern.replace(NUMBER_TOKEN, number);
    }

    render() {
        const selected = this.widget("selected_video")?.value ?? "";
        const selectedVideo = this.videos.find((v) => v.path === selected);
        this.latentLabel.textContent = selectedVideo
            ? `Latent: ${this.latentFor(selectedVideo.number)}`
            : selected ? `Selected (not in list): ${selected}` : "No video selected";

        this.list.replaceChildren();
        for (const v of this.videos) {
            const isSel = v.path === selected;
            const row = el("div", {
                display: "flex", gap: "4px", alignItems: "center", padding: "2px",
                borderRadius: "3px", background: isSel ? "rgba(80,140,255,0.35)" : "rgba(255,255,255,0.05)",
            });
            if (this.showThumbs) {
                const thumb = el("video", { width: "96px", height: "54px", objectFit: "cover", background: "#000", flexShrink: "0", cursor: "pointer" });
                thumb.muted = true;
                thumb.preload = "metadata";
                thumb.src = `${videoUrl(v)}#t=0.1`;
                thumb.addEventListener("click", () => this.play(v));
                row.append(thumb);
            }
            const name = el("span", { flex: "1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, v.path);
            name.title = v.path;
            const num = el("span", { opacity: "0.7" }, `#${v.number}`);
            row.append(
                name,
                num,
                button("Play", () => this.play(v)),
                button(isSel ? "Unselect" : "Select", () => this.select(v)),
                button("Delete", () => this.deleteVideo(v), { color: "#ff6b6b" })
            );
            this.list.append(row);
        }
    }
}

app.registerExtension({
    name: "martinodes.VideoLatentPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const picker = new Picker(this);
            this.martinodesPicker = picker;
            pickers.add(picker);

            this.addDOMWidget("video_list", "martinodes_video_list", picker.root, {
                serialize: false,
                getMinHeight: () => 200,
            });

            let timer;
            const regexWidget = picker.widget("video_regex");
            if (regexWidget) {
                const cb = regexWidget.callback;
                regexWidget.callback = function () {
                    const res = cb?.apply(this, arguments);
                    clearTimeout(timer);
                    timer = setTimeout(() => picker.refresh(), 400);
                    return res;
                };
            }
            for (const name of ["latent_pattern", "selected_video"]) {
                const w = picker.widget(name);
                if (!w) continue;
                const cb = w.callback;
                w.callback = function () {
                    const res = cb?.apply(this, arguments);
                    picker.render();
                    return res;
                };
            }

            this.setSize([Math.max(this.size[0], 480), Math.max(this.size[1], 400)]);
            picker.refresh();
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            // Widget values are restored after creation; reload with the saved regex
            this.martinodesPicker?.refresh();
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.martinodesPicker) {
                this.martinodesPicker.closePlayer();
                pickers.delete(this.martinodesPicker);
            }
            return onRemoved?.apply(this, arguments);
        };
    },
});
