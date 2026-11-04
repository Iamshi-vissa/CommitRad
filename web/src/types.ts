export type Component = "off_hours" | "risk" | "burstiness" | "negativity";

export interface Reason {
  feature: string;
  value: number;
  shap: number;
  effect: "raises" | "lowers";
  text: string;
}

export interface Commit {
  hash: string;
  author: string;
  author_email: string;
  timestamp: string;
  subject: string;
  risk: number;
  risk_percentile: number;
  via_pr: boolean;
  off_hours: boolean;
  bug_introducing: boolean;
  top_reasons: Reason[];
  repo?: string;
  repo_url?: string;
}

export interface StrainWeek {
  author?: string;
  author_email?: string;
  week: string;
  n_commits: number;
  strain: number;
  components: Record<Component, number>;
}

export interface RepoResult {
  repo: string;
  slug: string;
  head: string;
  n_commits: number;
  n_bug_introducing_szz: number;
  model: { label_mode: string; test_pr_auc: number; test_roc_auc: number; in_training_data: boolean };
  commits: Commit[];
  strain: StrainWeek[];
  disclaimer: string;
  snapshot_at?: string;
  cached?: boolean;
}

export interface RepoSummary {
  full_name: string;
  url: string;
  fork: boolean;
  private: boolean;
  language: string | null;
  parent: string | null;
  ahead_by: number | null;
  behind_by: number | null;
  status: string;
  error: string | null;
  my_commits: number;
  repo_commits: number;
  my_bug_rate: number | null;
  repo_bug_rate: number | null;
  my_via_pr: number | null;
  my_off_hours: number | null;
  my_risk_pct: number | null;
  repo_risk_pct: number | null;
  in_training_data: boolean;
}

export interface Push {
  at: string;
  weekday: number;
  hour: number;
  repo: string;
  branch: string;
  commits: number;
}

export interface AccountResult {
  login: string;
  name: string | null;
  avatar_url: string | null;
  profile_url: string;
  repos: RepoSummary[];
  commits: Commit[];
  strain: StrainWeek[];
  pushes: Push[];
  totals: {
    repos_analyzed: number;
    forks: number;
    my_commits: number;
    my_bug_introducing: number;
    my_bug_rate: number | null;
    my_via_pr: number | null;
    my_off_hours: number | null;
    my_risk_pct: number | null;
    pushes_90d: number;
    push_commits_90d: number;
  };
  emails_matched: number;
  api_calls: number;
}
