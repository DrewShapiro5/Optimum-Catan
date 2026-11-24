import type { CSSProperties } from "react";

export type IconName =
  | "WOOD"
  | "BRICK"
  | "SHEEP"
  | "WHEAT"
  | "ORE"
  | "DESERT"
  | "settlement"
  | "city"
  | "road"
  | "person"
  | "army"
  | "bank"
  | "trade"
  | "dev"
  | "next"
  | "settings"
  | "help"
  | "expand"
  | "close"
  | "check"
  | "robber"
  | "back";

export function Artwork({ name }: { name: IconName }) {
  switch (name) {
    case "WOOD":
      return (
        <>
          <path d="M29 43h7v17h-7z" fill="#79522c" />
          <path
            d="M32 3 19 20h7L14 34h9L8 49h48L41 34h9L38 20h7Z"
            fill="#409627"
            stroke="#245326"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path d="m32 12-7 10m7 5-10 10" stroke="#69ba3c" strokeWidth="2" />
        </>
      );
    case "BRICK":
      return (
        <g stroke="#8f4f34" strokeWidth="1.7" fill="#ffe0b0">
          <rect x="5" y="11" width="25" height="12" rx="2" />
          <rect x="33" y="11" width="25" height="12" rx="2" />
          <rect x="1" y="26" width="17" height="12" rx="2" />
          <rect x="21" y="26" width="25" height="12" rx="2" />
          <rect x="49" y="26" width="14" height="12" rx="2" />
          <rect x="6" y="41" width="25" height="12" rx="2" />
          <rect x="34" y="41" width="25" height="12" rx="2" />
        </g>
      );
    case "SHEEP":
      return (
        <g stroke="#586963" strokeWidth="1.7" strokeLinejoin="round">
          <path d="M15 41v13h6l2-13m13 0 2 13h6V39" fill="#8b9691" />
          <path
            d="M10 31c-5-5-2-13 5-13 0-7 8-11 13-6 5-5 13-2 14 4 8-1 13 6 9 12 6 6 2 14-5 14-3 7-12 8-17 3-6 5-14 2-15-4-7 0-10-6-4-10Z"
            fill="#fffdf4"
          />
          <g transform="rotate(-25 44 23)">
            <path d="M44 19q-5-4-9-1 1 6 9 6" fill="#adb7ad" />
            <path
              d="M47 17c7-1 12 4 12 10v7c0 5-8 7-11 2l-4-9c-2-5-1-9 3-10Z"
              fill="#aab4aa"
            />
            <path
              d="M43 20c-3-2-2-6 1-6-1-4 4-6 6-3 3-3 7-1 6 2 4 0 6 4 3 6-2 2-5 1-6 0-3 3-7 3-10 1Z"
              fill="#fffdf4"
              strokeWidth="1.4"
              transform="rotate(15 50 18)"
            />
            <circle cx="53" cy="26" r="1.4" fill="#263f40" stroke="none" />
          </g>
          <path
            d="M16 26q3-4 7-1m6-5q4-3 7 1M22 38q4 3 7 0"
            fill="none"
            stroke="#d9dfd2"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </g>
      );
    case "WHEAT":
      return (
        <g
          stroke="#956822"
          strokeWidth="1.3"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          {[
            { x: 24, y: 52, angle: -23, scale: 0.7 },
            { x: 26, y: 58, angle: 16, scale: 0.83 },
            { x: 30, y: 61, angle: 39, scale: 0.65 },
          ].map((ear, i) => (
            <g
              key={i}
              transform={`translate(${ear.x} ${ear.y}) rotate(${ear.angle}) scale(${ear.scale})`}
            >
              <path d="M0 0Q-1-24 0-48" fill="none" />
              <path d="M0-46c-5-5-3-10 1-13 4 4 4 9-1 13Z" fill="#f5ce64" />
              {[-39, -28, -17].map((y, j) => (
                <g key={y} fill={j % 2 ? "#e8b640" : "#f0c552"}>
                  <path d={`M-1 ${y}c-9-2-12-7-11-13 8 2 11 7 11 13Z`} />
                  <path d={`M1 ${y + 3}c8-3 12-8 10-14-7 3-10 8-10 14Z`} />
                </g>
              ))}
            </g>
          ))}
        </g>
      );
    case "ORE":
      return (
        <g stroke="#5d8582" strokeWidth="1.8" fill="#d1eae0">
          <path d="m4 30 8-12 15 2 4 16-18 7Z" />
          <path d="m30 22 10-9 18 5-1 14-16 6Z" />
          <path d="m18 45 12-15 17 5 4 15-19 9Z" />
          <path d="m2 48 10-7 9 5-4 10H6Z" />
          <path
            d="m12 18 4 14 15 4m9-23 2 14 15 5M30 30l3 16 18 4"
            fill="none"
            stroke="#9abfb3"
          />
        </g>
      );
    case "DESERT":
      return (
        <g stroke="#6b702e" strokeWidth="2" fill="#9aa44c">
          <path d="M28 60V14c0-11 12-11 12 0v20h7V23c0-7 8-7 8 0v15c0 7-7 8-15 8v14ZM28 41H17C8 41 8 34 8 31V19c0-7 8-7 8 0v11h12Z" />
          <path
            d="M32 56V15m4 2v40M12 20v13m40-9v14"
            fill="none"
            strokeWidth="1"
          />
        </g>
      );
    case "settlement":
      return (
        <g stroke="#343b46" strokeWidth="2.5" strokeLinejoin="round">
          <path d="m9 28 23-23 23 23v29H9Z" fill="currentColor" />
          <path
            d="m9 28 23-23v31L9 47Z"
            fill="#fff"
            opacity=".2"
            stroke="none"
          />
          <path d="m9 28 23 8 23-8M32 5v31" fill="none" />
          <path d="M28 57V45q5-6 10 0v12" fill="#293f4a" stroke="none" />
        </g>
      );
    case "city":
      return (
        <g stroke="#343b46" strokeWidth="2.5" strokeLinejoin="round">
          <path d="M5 28 23 7l18 21v7h17v23H5Z" fill="currentColor" />
          <path
            d="m5 28 18-21v31L5 45Z"
            fill="white"
            opacity=".24"
            stroke="none"
          />
          <path
            d="m5 28 18 10 18-10M23 7v31m18-3 17 11H35L23 38v20"
            fill="none"
          />
          <path d="M11 49h6v9h-6Zm29 1h6v8h-6Z" fill="#293f4a" stroke="none" />
        </g>
      );
    case "road":
      return (
        <g stroke="#343b46" strokeWidth="2" strokeLinejoin="round">
          <path d="m25 5 12-3 5 5v50l-12 5-5-5Z" fill="currentColor" />
          <path d="m25 5 5 5v52m0-52 12-3" fill="none" />
          <path
            d="m26 7 3 4v46l-3-2Z"
            fill="#fff"
            opacity=".28"
            stroke="none"
          />
        </g>
      );
    case "person":
      return (
        <g fill="#fffdf5" stroke="#294b5a" strokeWidth="2">
          <path d="M16 54V42c0-15 32-15 32 0v12q-16 10-32 0Z" />
          <ellipse cx="32" cy="20" rx="12" ry="15" />
        </g>
      );
    case "army":
      return (
        <>
          <g opacity=".75" transform="translate(-8 -4) scale(.8)">
            <Artwork name="person" />
          </g>
          <g opacity=".75" transform="translate(24 -4) scale(.8)">
            <Artwork name="person" />
          </g>
          <g transform="translate(7 12) scale(.8)">
            <Artwork name="person" />
          </g>
        </>
      );
    case "bank":
      return (
        <g stroke="#8c6b3c" strokeWidth="1.8" fill="#e3c38b">
          <path d="M4 23 32 7l28 16v6H4Zm3 29h50v5H7Zm-3 6h56v5H4Z" />
          <path d="M11 30h8v21h-8Zm17 0h8v21h-8Zm17 0h8v21h-8Z" />
          <circle cx="32" cy="20" r="8" fill="#f6e4b9" />
          <text
            x="32"
            y="24"
            textAnchor="middle"
            fill="#8c6b3c"
            stroke="none"
            fontSize="11"
            fontWeight="bold"
          >
            C
          </text>
        </g>
      );
    case "trade":
      return (
        <g stroke="#244b5b" strokeWidth="2" fill="#e4f8fb">
          <path d="M51 23C43 10 25 9 15 18l7 7H3V6l7 7C26-1 51 4 59 19Z" />
          <path d="M13 41c8 13 26 14 36 5l-7-7h19v19l-7-7C38 65 13 60 5 45Z" />
        </g>
      );
    case "dev":
      return (
        <g stroke="#344e59" strokeWidth="2.5">
          <rect x="10" y="3" width="44" height="58" rx="3" fill="#9870bd" />
          <rect x="14" y="7" width="36" height="50" rx="1" fill="#7960b5" />
          <circle cx="32" cy="32" r="17" fill="#a8db8b" stroke="#effff3" />
          <path d="m22 44 18-24 5 4-18 24Z" fill="#ae6e2d" />
          <path d="m30 17 9-5 13 10-5 8-9-5-3 5-8-6Z" fill="#b9d7db" />
        </g>
      );
    case "next":
      return (
        <g
          fill="#d5effb"
          stroke="#25495b"
          strokeWidth="3.5"
          strokeLinejoin="round"
        >
          <path d="m8 9 23 23L8 55Zm25 0 23 23-23 23Z" />
        </g>
      );
    case "robber":
      return (
        <g stroke="#415866" strokeWidth="2" fill="#aab6b9">
          <ellipse cx="32" cy="52" rx="15" ry="7" />
          <path d="M19 48c-3-13 1-26 13-26s16 13 13 26q-13 12-26 0Z" />
          <circle cx="32" cy="16" r="12" />
        </g>
      );
    case "check":
      return (
        <path
          d="m10 33 14 15L55 14l-8-7-24 27-7-8Z"
          fill="#fffdf5"
          stroke="#294b5a"
          strokeWidth="2"
        />
      );
    case "close":
      return (
        <path
          d="m14 7 18 18L50 7l7 7-18 18 18 18-7 7-18-18-18 18-7-7 18-18L7 14Z"
          fill="#fffdf5"
          stroke="#294b5a"
          strokeWidth="2"
        />
      );
    case "settings":
      return (
        <g stroke="#294b5a" strokeWidth="3" fill="#d8f4fa">
          <path d="m24 3 16 0 2 10 10-3 8 14-8 7 8 8-8 14-10-3-3 11H24l-2-11-10 3L4 39l8-8-8-7 8-14 10 3Z" />
          <circle cx="32" cy="32" r="15" />
          <circle cx="32" cy="32" r="7" fill="#1680b4" />
        </g>
      );
    case "help":
      return (
        <g stroke="#294b5a" strokeWidth="2" fill="#daf4fa">
          <path d="M4 12q15-8 28 0 13-8 28 0v42q-14-8-28 0-14-8-28 0Z" />
          <path
            d="M32 12v42M10 19q8-4 16 0m-16 8q8-4 16 0m12-8q8-4 16 0m-16 8q8-4 16 0"
            fill="none"
          />
        </g>
      );
    case "expand":
      return (
        <path
          d="M23 6H6v17m0-17 19 19m16-19h17v17m0-17L39 25M6 41v17h17M6 58l19-19m33 2v17H41m17 0L39 39"
          fill="none"
          stroke="#d8f4fa"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      );
    case "back":
      return (
        <path
          d="M36 11 15 32l21 21M16 32h39"
          fill="none"
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      );
  }
}

export default function Icon({
  name,
  className = "",
  style,
}: {
  name: IconName;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <svg
      className={`game-icon ${className}`}
      style={style}
      viewBox="0 0 64 64"
      aria-hidden="true"
    >
      <Artwork name={name} />
    </svg>
  );
}
