interface SwitchProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  title?: string; // additional switch tooltip
}

export function Switch({ label, checked, onChange, disabled, title }: SwitchProps) {
  return (
    <label className={`switch-row${disabled ? " switch-row--disabled" : ""}`} title={title}>
      <span className="switch-label">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        className={`switch-track${checked ? " switch-track--on" : ""}`}
        onClick={() => onChange(!checked)}
      >
        <span className="switch-thumb" />
      </button>
    </label>
  );
}
