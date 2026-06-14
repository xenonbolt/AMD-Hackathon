export interface JavaFile {
  name: string;
  path: string;
  content: string;
  originalContent: string;
}

export interface Vulnerability {
  id: string;
  type: string;
  severity: 'High' | 'Medium' | 'Low';
  filePath: string;
  lineNumber: number;
  snippet: string;
  description: string;
  recommendation: string;
  status: 'Scanned' | 'Fixing' | 'Diff Ready' | 'Approved' | 'Rejected' | 'Ignored';
  remediatedSnippet?: string;
  remediationExplanation?: string;
}

export interface ProjectDemo {
  name: string;
  description: string;
  files: JavaFile[];
}
