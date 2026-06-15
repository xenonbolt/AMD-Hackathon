export interface JavaFile {
  name: string;
  path: string;
  content: string;
  originalContent: string;
}

export interface Vulnerability {
  id?: string; // id is sometimes missing from JSON
  cwe_id?: string;
  cwe_name?: string;
  confidence?: number;
  location?: { line: number };
  impact?: string;
  type?: string;
  severity: string;
  filePath: string;
  lineNumber: number;
  snippet?: string;
  description: string;
  recommendation: string;
  status: string;
  remediatedSnippet?: string;
  remediationExplanation?: string;
  fullRemediatedContent?: string;
}

export interface ScanHistoryEntry {
  id: string;
  timestamp: Date;
  mode: string;
  files: JavaFile[];
  vulnerabilities: Vulnerability[];
}

export interface ProjectDemo {
  name: string;
  description: string;
  files: JavaFile[];
}
