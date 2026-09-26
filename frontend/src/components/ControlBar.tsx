interface Props {
  episodeId: string | null;
  ended: boolean;
  onCreateEpisode: (seed: number) => void;
  onStep: (n: number) => void;
  onInjectIncident: (severity: number) => void;
}

/** Not one of dev doc §11.3's own four named panes — a small, disclosed
 * addition: §11.2's `POST /episodes/{id}/step` is explicitly "the demo
 * driver", implying an operator controls stepping; this is that
 * control surface. */
export function ControlBar({ episodeId, ended, onCreateEpisode, onStep, onInjectIncident }: Props) {
  return (
    <div className="control-bar">
      <button onClick={() => onCreateEpisode(Math.floor(Math.random() * 1_000_000))}>
        New episode
      </button>
      {episodeId && <span className="episode-id">episode: {episodeId}</span>}
      <button disabled={!episodeId || ended} onClick={() => onStep(1)}>
        Step ×1 tick
      </button>
      <button disabled={!episodeId || ended} onClick={() => onStep(3)}>
        Step ×3 (1 decision)
      </button>
      <button disabled={!episodeId || ended} onClick={() => onInjectIncident(0.9)}>
        Inject severe flood
      </button>
      {ended && <span className="episode-ended-badge">episode ended</span>}
    </div>
  );
}
