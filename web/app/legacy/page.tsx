export default function LegacyPage() {
  const url = process.env.NEXT_PUBLIC_LEGACY_UI_URL ?? "http://localhost:8501";
  return <main className="shell"><header><a className="logo" href="/">N</a><span className="eyebrow">LEGACY PROTOTYPES</span></header>
    <section className="intro"><p className="eyebrow">ARCHIVED EXPERIENCE</p><h1>Legacy<br /><em>workbench.</em></h1>
      <p className="subhead">The Streamlit prototype is available as a separate service when the legacy Compose profile is enabled.</p>
      <a className="primary" href={url}>Open legacy Streamlit UI <span>↗</span></a>
    </section>
  </main>;
}
