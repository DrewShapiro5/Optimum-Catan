import Icon from "./Icons";
import type { ResourceCard } from "../utils/api.types";
import { LABELS } from "./model";

export default function Card({
  resource,
  count,
  small = false,
  onClick,
  disabled = false,
}: {
  resource: ResourceCard | "dev" | "hidden";
  count?: number;
  small?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}) {
  const label =
    resource === "hidden"
      ? "Resource cards"
      : resource === "dev"
        ? "Development cards"
        : LABELS[resource];
  const content = (
    <>
      <span className="card-inset">
        {resource === "hidden" ? <b>?</b> : <Icon name={resource} />}
      </span>
      {count !== undefined && <span className="card-count">{count}</span>}
    </>
  );
  const className = `resource-card card-${resource.toLowerCase()} ${small ? "small" : ""}`;
  return onClick ? (
    <button
      className={className}
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={`${label}${count !== undefined ? `: ${count}` : ""}`}
      title={label}
    >
      {content}
    </button>
  ) : (
    <span
      className={className}
      title={`${label}${count !== undefined ? `: ${count}` : ""}`}
      aria-label={`${label}${count !== undefined ? `: ${count}` : ""}`}
    >
      {content}
    </span>
  );
}
