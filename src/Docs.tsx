import { useState } from "react";

type DocKey = "image" | "audio" | "video" | "imageUpscale" | "videoUpscale";

const SECTIONS: Array<{ id: DocKey; label: string }> = [
  { id: "image", label: "Image Creation" },
  { id: "audio", label: "Music & Sound" },
  { id: "video", label: "Video Creation" },
  { id: "imageUpscale", label: "Image Upscale" },
  { id: "videoUpscale", label: "Video Upscale" },
];

type ModelDoc = {
  name: string;
  bestFor: string;
  startingSettings: string[];
  watch: string;
};

const IMAGE_MODELS: ModelDoc[] = [
  {
    name: "Ideogram 4 NF4",
    bestFor:
      "Realistic humans, portraits, natural skin, direct eye contact, and scenes where a clean photographic read matters more than raw speed.",
    startingSettings: [
      "Architecture: Ideogram 4 (open weights).",
      "Use the NF4 local model as the normal option. The FP8 entry is guarded because it has hard-crashed during local smoke tests.",
      "Magic prompt: on, Local mode, unless you intentionally want raw JSON or exact prompt control.",
      "Speed: High for normal work. Turbo 12 for drafts. Standard 20 for balanced quality. Quality 48 only for final attempts.",
      "Portrait: start at 768 x 1152. Move to 1024 x 1536 when the composition is already working.",
    ],
    watch:
      "Cold loads are slow and are isolated on purpose so a bad run is less likely to take down the whole app. Avoid batching huge counts until one image has passed.",
  },
  {
    name: "Z-Image Turbo",
    bestFor:
      "Fast, aesthetically pleasing images with strong overall taste. It is a good first-pass model when you want attractive output quickly.",
    startingSettings: [
      "Architecture: Z-Image (components).",
      "CFG: 1.0.",
      "Steps: 8 for turbo drafts, 20-28 when you want to spend more time on polish.",
      "Use the detected Z-Image VAE and text encoder when the app auto-selects them.",
      "Portrait: 768 x 1152 for speed, 1024 x 1536 for keepers.",
    ],
    watch:
      "It is often the best-looking fast option, but exact prompt adherence can lag behind FLUX or Ideogram on complex scenes.",
  },
  {
    name: "FLUX.1 Dev",
    bestFor:
      "Prompt adherence, general-purpose composition, reliable non-human subjects, and scenes that need clear structure.",
    startingSettings: [
      "Architecture: FLUX1 (components).",
      "CFG: 1.0. In this UI, 1 disables KSampler-style CFG for FLUX.",
      "Steps: 28 for normal quality. Use 4 only for Schnell/distilled models.",
      "Use the detected FLUX VAE, CLIP-L, and T5 encoder selections.",
      "Portrait: 768 x 1152 to test, 1024 x 1536 for stronger detail.",
    ],
    watch:
      "Human portraits can look plastic or oily. Ask for natural skin texture, soft window light, and minimal retouching rather than pushing CFG higher.",
  },
  {
    name: "Krea 2 Raw FP8",
    bestFor:
      "Experimental stylized portraits, cinematic color, and cases where you want the Krea look available as an option.",
    startingSettings: [
      "Architecture: Krea 2 (open weights).",
      "Use Apply Krea Reddit Stack for the tested baseline.",
      "Stack: Krea 2 Raw FP8 scaled, WAN 2.1 VAE, turbo LoRA at 0.60, filter bypass LoRA at 1.00.",
      "Sampling: 12 steps, CFG 1.0, portrait 768 x 1152.",
      "LoRA slots are additive. Keep the stack to two LoRAs first, then add a third only for a targeted test.",
    ],
    watch:
      "The current Krea result can look filtered or overprocessed. The app fails fast on missing or incompatible Krea LoRAs so bad stacks do not waste a long run.",
  },
];

export default function Docs({ active }: { active: boolean }) {
  const [section, setSection] = useState<DocKey>("image");

  return (
    <main className={"pane docs-root" + (active ? "" : " tab-hidden")}>
      <div className="docs-header">
        <div>
          <div className="section-title">Help</div>
          <h1>Kraken Art Documentation</h1>
          <p className="muted">
            Practical settings and workflow notes for the tools wired into this app.
          </p>
        </div>
        <div className="docs-tabs" aria-label="Documentation sections">
          {SECTIONS.map((item) => (
            <button
              key={item.id}
              className={section === item.id ? "active" : ""}
              onClick={() => setSection(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {section === "image" && <ImageCreationDocs />}
      {section === "audio" && <AudioDocs />}
      {section === "video" && <VideoDocs />}
      {section === "imageUpscale" && <ImageUpscaleDocs />}
      {section === "videoUpscale" && <VideoUpscaleDocs />}
    </main>
  );
}

function ImageCreationDocs() {
  return (
    <div className="docs-content">
      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Image Creation</h2>
          <p className="muted">
            Pick the model for the job first, then tune dimensions, steps, LoRAs, and seed. Use Output name to label tests before generating.
          </p>
        </div>
        <div className="docs-callout">
          For portrait tests, start at 768 x 1152 with count 1. Keep the seed when comparing models. Name outputs with the model or stack name so the Library stays readable.
        </div>
      </section>

      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Model Guide</h2>
          <p className="muted">Current local guidance based on the app wiring and recent smoke tests.</p>
        </div>
        <div className="model-doc-grid">
          {IMAGE_MODELS.map((model) => (
            <article key={model.name} className="model-doc">
              <h3>{model.name}</h3>
              <p>{model.bestFor}</p>
              <h4>Best starting settings</h4>
              <ul>
                {model.startingSettings.map((item) => <li key={item}>{item}</li>)}
              </ul>
              <h4>Watch for</h4>
              <p className="muted">{model.watch}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Settings That Matter</h2>
          <p className="muted">The controls that most often change quality, speed, or failure risk.</p>
        </div>
        <div className="docs-table-wrap">
          <table className="docs-table">
            <thead>
              <tr>
                <th>Control</th>
                <th>Use it for</th>
                <th>Starting point</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Output name</td>
                <td>Manual filename prefix for experiments and keeper runs.</td>
                <td>Use names like ideogram4-portrait-test or krea2-reddit-stack.</td>
              </tr>
              <tr>
                <td>Dimensions</td>
                <td>Composition and native detail.</td>
                <td>768 x 1152 for portrait tests, 1024 x 1536 for higher-detail keepers.</td>
              </tr>
              <tr>
                <td>LoRAs</td>
                <td>Model-specific style, speed, realism, or bypass behavior.</td>
                <td>Use one or two slots first. Krea Reddit Stack uses two LoRAs by design.</td>
              </tr>
              <tr>
                <td>CFG</td>
                <td>Prompt guidance on SDXL-style models. Flow models often treat it differently.</td>
                <td>SDXL around 7. FLUX, Z-Image, and Krea turbo stacks usually at 1.</td>
              </tr>
              <tr>
                <td>Steps</td>
                <td>Quality versus time.</td>
                <td>Turbo models: 8-12. FLUX/Z-Image: 20-28. Ideogram: 12/20/48 presets.</td>
              </tr>
              <tr>
                <td>Seed</td>
                <td>Repeatability and model comparisons.</td>
                <td>Turn off random seed when comparing settings or models against one prompt.</td>
              </tr>
              <tr>
                <td>Upscale</td>
                <td>Fast ESRGAN enlargement or slower SDXL tile refinement.</td>
                <td>Use ESRGAN for quick keepers. Use USDU only when the source is already good.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Robustness Notes</h2>
        </div>
        <ul className="docs-list">
          <li>Guarded models are shown but not selected by default. Ideogram FP8 is guarded because it has caused Windows access violations during local testing.</li>
          <li>Bad Krea LoRA paths, incompatible formats, and unsupported sampler stacks should fail before a long generation starts.</li>
          <li>Use Clear VRAM before switching between large image, audio, and video jobs when memory is tight.</li>
          <li>When testing quality, change one variable at a time: model, LoRA stack, seed, dimensions, or steps.</li>
        </ul>
      </section>
    </div>
  );
}

function AudioDocs() {
  return (
    <div className="docs-content">
      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Music & Sound Creation</h2>
          <p className="muted">
            The Music tab covers songs, instrumentals, sound effects, speech, and multi-voice dialogue.
          </p>
        </div>
        <ul className="docs-list">
          <li>Music with vocals uses ACE-Step and Song Studio when available. Use lyrics for structure and the style prompt for genre, mood, instrumentation, and vocal direction.</li>
          <li>Instrumental music uses Stable Audio 3. Keep prompts acoustic and concrete, then set duration before adding complexity.</li>
          <li>Sound effects and ambience work best as short descriptions. MOSS-SoundEffect is strongest around 1-30 seconds; use more steps for higher quality.</li>
          <li>Speech can use MOSS-TTS for direct narration or LuxTTS for reference-voice cloning. Dialogue maps each bracketed speaker line to a voice.</li>
          <li>Export selected songs to MP3 from the library when you need a portable file with embedded cover art.</li>
        </ul>
      </section>
    </div>
  );
}

function VideoDocs() {
  return (
    <div className="docs-content">
      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Video Creation</h2>
          <p className="muted">
            The Video tab is wired around WAN 2.2 text-to-video and image-to-video workflows.
          </p>
        </div>
        <ul className="docs-list">
          <li>Image-to-video is the safer path for character and product consistency. Choose a strong still image, then describe only the motion you want.</li>
          <li>Text-to-video is better for exploratory motion ideas where exact identity is less important.</li>
          <li>WAN speed LoRAs are intended for short-step sampling. Leave them enabled for 4-step drafts, disable them for full 20-40 step sampling.</li>
          <li>Use the motion presets for a baseline, then edit prompt, frames, FPS, steps, and CFG. The app snaps frame counts to WAN-friendly multiples.</li>
          <li>Use the last frame from a result as a new image-to-video input when building longer sequences.</li>
        </ul>
      </section>
    </div>
  );
}

function ImageUpscaleDocs() {
  return (
    <div className="docs-content">
      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Image Upscale</h2>
          <p className="muted">
            Use this tab when the image is already selected and you want more control than the quick upscale inside Image Creation.
          </p>
        </div>
        <ul className="docs-list">
          <li>ESRGAN is the fast single-pass path. It preserves the source better and is the right first test for most keepers.</li>
          <li>USDU runs tile plus img2img refinement through an SDXL model. It is slower and can add detail, but it can also change faces and fine identity.</li>
          <li>For USDU, start with 20 steps, denoise 0.20, CFG 6, and automatic tile size. Raise denoise only when you want visible repainting.</li>
          <li>Use target resolution when you know the final size. Use factor mode when preserving a clean multiplier matters.</li>
          <li>If memory fails at high resolution, lower tile size before changing the creative settings.</li>
        </ul>
      </section>
    </div>
  );
}

function VideoUpscaleDocs() {
  return (
    <div className="docs-content">
      <section className="docs-panel">
        <div className="docs-panel-head">
          <h2>Video Upscale</h2>
          <p className="muted">
            Choose preservation or reconstruction first. That decision matters more than any single slider.
          </p>
        </div>
        <ul className="docs-list">
          <li>ESRGAN preserves source motion, identity, and composition. It is the default for speed and predictable results.</li>
          <li>SeedVR2 reconstructs detail and can look more cinematic, but it may alter faces, logos, fabric, and small background details.</li>
          <li>Start with HD/1080p for a test clip, then move to 2K or 4K once temporal stability is acceptable.</li>
          <li>For SeedVR2, low AI detail strength keeps more of the source. Higher strength increases polish and hallucination risk.</li>
          <li>Use RIFE Vulkan interpolation when you need smoother output. Disable interpolation when preserving exact source timing matters.</li>
          <li>Use VAE tiling or ESRGAN tile sizes of 768 or 1024 when 2K/4K runs hit memory limits.</li>
        </ul>
      </section>
    </div>
  );
}
