import Link from "next/link";

import { RedTeamPanel } from "@/components/redteam/RedTeamPanel";

export default function RedTeamPage() {
  return (
    <main className="flex flex-1 flex-col gap-6 p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            Red-team the gate
          </h1>
          <p className="text-sm text-muted-foreground">
            Adversarial battery — does the action classifier hold under
            pressure?
          </p>
        </div>
        <Link
          href="/"
          className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          ← Agent
        </Link>
      </header>

      <RedTeamPanel />
    </main>
  );
}
