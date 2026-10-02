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

function button(label, onClick) {
    const b = el("button", { fontSize: "11px", padding: "1px 6px", cursor: "pointer", whiteSpace: "nowrap" }, label);
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

        this.player = el("div", { display: "none", flexDirection: "column", gap: "2px" });
        this.playerVideo = el("video", { width: "100%", maxHeight: "240px", background: "#000" });
        this.playerVideo.controls = true;
        this.playerTitle = el("span", { flex: "1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
        const playerBar = el("div", { display: "flex", gap: "4px", alignItems: "center" });
        playerBar.append(this.playerTitle, button("Close", () => this.closePlayer()));
        this.player.append(playerBar, this.playerVideo);

        this.list = el("div", { flex: "1", overflowY: "auto", display: "flex", flexDirection: "column", gap: "2px", minHeight: "0" });

        this.root.append(toolbar, this.latentLabel, this.player, this.list);
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
        this.playerTitle.textContent = v.path;
        this.playerVideo.src = videoUrl(v);
        this.player.style.display = "flex";
        this.playerVideo.play().catch(() => {});
    }

    closePlayer() {
        this.playerVideo.pause();
        this.playerVideo.removeAttribute("src");
        this.playerVideo.load();
        this.player.style.display = "none";
    }

    select(v) {
        const w = this.widget("selected_video");
        if (!w) return;
        w.value = v.path;
        w.callback?.(w.value);
        this.render();
        this.node.setDirtyCanvas(true, true);
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
            row.append(name, num, button("Play", () => this.play(v)), button(isSel ? "Selected" : "Select latent", () => this.select(v)));
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
