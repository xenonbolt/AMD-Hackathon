import React, { useState, useRef, useEffect, useMemo } from "react";
import {
  Shield,
  FileCode,
  FolderOpen,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Flame,
  Eye,
  RefreshCw,
  Check,
  Trash2,
  HelpCircle,
  GitCommit,
  Terminal,
  ArrowRight,
  ChevronRight,
  Sparkles,
  Code2,
  FileCheck,
  AlertOctagon,
  KeyRound,
  FileSpreadsheet,
  Info,
  ChevronDown,
  ChevronUp,
  LayoutDashboard,
  History
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { JavaFile, Vulnerability, ScanHistoryEntry } from "./types";

// Lightweight regex-based syntax tokenizer for Java code
function tokenizeJavaLine(line: string): { text: string; type: string }[] {
  if (!line) return [{ text: " ", type: "text" }];

  const regex = /(\/\/.*)|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])*')|(@[a-zA-Z0-9_]+)|(\b(?:public|private|protected|class|interface|enum|extends|implements|import|package|return|new|if|else|for|while|try|catch|finally|throw|throws|final|static|void|int|double|float|long|boolean|char|short|byte|null|true|false|this|super)\b)|(\b\d+(?:\.\d+)?\b)|(\b[A-Z][a-zA-Z0-9_]*\b)|(\b[a-z_][a-zA-Z0-9_]*(?=\s*\())|([+\-*/%&=<>!|~^:?(){}[\];.,]+)|(\s+)|([^\s]+)/g;

  const tokens: { text: string; type: string }[] = [];
  let match;

  while ((match = regex.exec(line)) !== null) {
    const [
      full,
      comment,
      stringDouble,
      stringSingle,
      annotation,
      keyword,
      number,
      className,
      methodName,
      operator,
      whitespace,
      other
    ] = match;

    if (comment !== undefined) {
      tokens.push({ text: comment, type: "comment" });
    } else if (stringDouble !== undefined) {
      tokens.push({ text: stringDouble, type: "string" });
    } else if (stringSingle !== undefined) {
      tokens.push({ text: stringSingle, type: "string" });
    } else if (annotation !== undefined) {
      tokens.push({ text: annotation, type: "annotation" });
    } else if (keyword !== undefined) {
      tokens.push({ text: keyword, type: "keyword" });
    } else if (number !== undefined) {
      tokens.push({ text: number, type: "number" });
    } else if (className !== undefined) {
      tokens.push({ text: className, type: "class-name" });
    } else if (methodName !== undefined) {
      tokens.push({ text: methodName, type: "method-name" });
    } else if (operator !== undefined) {
      tokens.push({ text: operator, type: "operator" });
    } else if (whitespace !== undefined) {
      tokens.push({ text: whitespace, type: "text" });
    } else {
      tokens.push({ text: other, type: "text" });
    }
  }

  return tokens;
}

// Map Java syntax elements to classic VS Code Dark style coloring with soft green hints
function renderHighlightedCode(line: string) {
  const tokens = tokenizeJavaLine(line);
  return (
    <>
      {tokens.map((token, i) => {
        let styleClass = "";
        switch (token.type) {
          case "comment":
            styleClass = "text-emerald-500/80 italic"; // VS Code green comments
            break;
          case "string":
            styleClass = "text-amber-300"; // Rich VS Code amber/yellow strings
            break;
          case "annotation":
            styleClass = "text-emerald-400 font-semibold"; // Green annotations
            break;
          case "keyword":
            styleClass = "text-sky-450 font-bold"; // Deep blue keywords
            break;
          case "number":
            styleClass = "text-emerald-400 font-semibold"; // Green numbers
            break;
          case "class-name":
            styleClass = "text-teal-300 font-semibold"; // Teal classes
            break;
          case "method-name":
            styleClass = "text-yellow-200"; // Warm yellow-gold methods
            break;
          case "operator":
            styleClass = "text-slate-400"; // Operators
            break;
          default:
            styleClass = "text-slate-200"; // Active default foreground text
        }
        return (
          <span key={i} className={styleClass}>
            {token.text}
          </span>
        );
      })}
    </>
  );
}

export default function App() {
  // Projects and files
  const [loadedFiles, setLoadedFiles] = useState<JavaFile[]>([]);
  const [activeFileIndex, setActiveFileIndex] = useState<number>(0);
  const [droppedItemName, setDroppedItemName] = useState<string | null>(null);

  // App Level Tabs
  const [activeTab, setActiveTab] = useState<"scan" | "history" | "rocm">("scan");
  const [scanHistory, setScanHistory] = useState<ScanHistoryEntry[]>([]);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState<boolean>(false);
  const [expandedHistoryIds, setExpandedHistoryIds] = useState<string[]>([]);
  const [rocmStats, setRocmStats] = useState({ vram: 14200, load: 45, cpu: 25, ram: 128, vramTotal: 198000, gpuName: 'AMD MI300X', gpuType: 'HBM3 Memory Cluster', modelsLoaded: ['Loading...'], apiHealth: 'Loading...' });

  const toggleHistoryExpanded = (id: string) => {
    setExpandedHistoryIds(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  // Load history from localstorage directory backend
  useEffect(() => {
    fetch('/local-api/history')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setScanHistory(data.map((item: any) => ({
            ...item,
            timestamp: new Date(item.timestamp)
          })));
        }
        setIsHistoryLoaded(true);
      })
      .catch(err => {
        console.error("Failed to load history from local server", err);
        setIsHistoryLoaded(true);
      });
  }, []);

  const isInitialMount = useRef(true);

  // Persist history to localstorage directory backend whenever it updates
  useEffect(() => {
    if (!isHistoryLoaded) return;
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    fetch('/local-api/history', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scanHistory)
    }).catch(err => console.error("Failed to save history to local server", err));
  }, [scanHistory, isHistoryLoaded]);

  // Poll ROCm Telemetry
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (activeTab === "rocm") {
      const fetchTelemetry = async () => {
        try {
          const res = await fetch("/api/telemetry");
          if (res.ok) {
            const data = await res.json();
            setRocmStats({
              vram: data.vram_usage || 14200,
              load: data.compute_load || 45,
              cpu: data.cpu_usage || 25,
              ram: data.ram_usage || 128,
              vramTotal: data.vram_total || 198000,
              gpuName: data.gpu_name || 'AMD MI300X',
              gpuType: data.gpu_type || 'HBM3 Memory Cluster',
              modelsLoaded: data.models_loaded || ['Qwen/Qwen2.5-Coder-7B-Instruct'],
              apiHealth: data.api_health || 'Unknown'
            });
          }
        } catch (e) {
          console.error("Failed to fetch ROCm telemetry", e);
        }
      };
      fetchTelemetry(); // Initial fetch
      interval = setInterval(fetchTelemetry, 1000); // Poll every 1 second
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeTab]);

  // Security elements
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([]);
  const [scanState, setScanState] = useState<"idle" | "scanning" | "completed">("idle");
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [scanProgressText, setScanProgressText] = useState<string>("");
  const [scanMode, setScanMode] = useState<"local_heuristics" | "gemini_ai" | "none">("none");
  const [scanMessage, setScanMessage] = useState<string>("");
  const [isSingleScanActive, setIsSingleScanActive] = useState<boolean>(false);

  // Specific remediation states (vulnerabilityId -> "idle" | "fixing" | "diff_ready" | "approved")
  const [remediatingIdState, setRemediatingIdState] = useState<Record<string, "idle" | "fixing" | "diff_ready" | "approved">>({});

  // Direct manual code pasting sandbox helper
  const [sandboxCode, setSandboxCode] = useState<string>("");
  const [sandboxFileName, setSandboxFileName] = useState<string>("MyVulnerableService.java");
  const [isSandboxOpen, setIsSandboxOpen] = useState<boolean>(false);

  // Drag and drop states
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Active highlighted lines in file editor (from selected vulnerability)
  const [selectedVulnerabilityId, setSelectedVulnerabilityId] = useState<string | null>(null);
  const activeFile = loadedFiles[activeFileIndex] || null;

  // Initialize data on load and auto-recover session
  useEffect(() => {
    fetch('/local-api/session')
      .then(res => res.json())
      .then(data => {
        if (data && Array.isArray(data.loadedFiles)) {
          setLoadedFiles(data.loadedFiles);
          setActiveFileIndex(data.activeFileIndex || 0);
          setVulnerabilities(data.vulnerabilities || []);
          setScanMode(data.scanMode || "none");
          setScanMessage(data.scanMessage || "");
          setScanState(data.scanState || "idle");

          if (data.scanState === "scanning") {
            // Re-trigger the scan based on the active state
            if (data.isSingleScanActive) {
              isRecoveringSingleScan.current = true;
            } else {
              isRecoveringGlobalScan.current = true;
            }
          }
        } else {
          setScanState("idle");
          setVulnerabilities([]);
        }
      })
      .catch(err => {
        console.error("Failed to recover session", err);
        setScanState("idle");
        setVulnerabilities([]);
      });
    setSelectedVulnerabilityId(null);
  }, []);

  const isRecoveringSingleScan = useRef(false);
  const isRecoveringGlobalScan = useRef(false);

  // Effect to retrigger scan if we recovered a scanning state
  useEffect(() => {
    if (loadedFiles.length > 0 && scanState === "scanning" && scanProgress === 0) {
      if (isRecoveringGlobalScan.current) {
        isRecoveringGlobalScan.current = false;
        triggerCodeScan();
      } else if (isRecoveringSingleScan.current && activeFile) {
        isRecoveringSingleScan.current = false;
        triggerSingleFileScan();
      }
    }
  }, [scanState, loadedFiles, activeFileIndex]);

  // Save session state on change
  useEffect(() => {
    if (loadedFiles.length === 0 && scanState === "idle") return; // Don't save empty initial state
    fetch('/local-api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        loadedFiles,
        activeFileIndex,
        scanState,
        vulnerabilities,
        scanMode,
        scanMessage,
        isSingleScanActive
      })
    }).catch(err => console.error("Failed to save session", err));
  }, [loadedFiles, activeFileIndex, scanState, vulnerabilities, scanMode, scanMessage, isSingleScanActive]);

  // Drag-and-drop overrides
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      processFileList(e.dataTransfer.files);
    }
  };

  const processFileList = async (files: FileList) => {
    const javaFilesList: JavaFile[] = [];
    const seenPaths = new Set<string>();

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith(".java") || file.type === "text/x-java-source") {
        const path = file.webkitRelativePath || `uploads/${file.name}`;
        
        // Skip common build and hidden directories to prevent duplicate scans
        if (
          path.includes("/target/") || 
          path.includes("/build/") || 
          path.includes("/bin/") || 
          path.includes("/.git/") ||
          path.includes("/.idea/") ||
          path.includes("/.vscode/")
        ) {
          continue;
        }

        if (seenPaths.has(path)) continue;
        seenPaths.add(path);

        const textContent = await readFileContent(file);
        javaFilesList.push({
          name: file.name,
          path: path,
          content: textContent,
          originalContent: textContent
        });
      }
    }

    if (javaFilesList.length > 0) {
      setLoadedFiles(javaFilesList);
      setActiveFileIndex(0);
      setScanState("idle");
      setVulnerabilities([]);
      setSelectedVulnerabilityId(null);

      // Determine dropped name based on common path or single file
      let droppedName = "Unknown";
      if (files.length === 1) {
        droppedName = files[0].name;
      } else if (files[0].webkitRelativePath) {
        droppedName = files[0].webkitRelativePath.split('/')[0];
      } else {
        droppedName = `${files.length} Files Loaded`;
      }
      setDroppedItemName(droppedName);
    } else {
      alert("No Java source files (.java) detected in the uploaded list.");
    }
  };

  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string || "");
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  };

  // Handle standard individual file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFileList(e.target.files);
    }
  };

  // Add a fully custom pasted Java code file inside workspace
  const handleAddSandboxFile = () => {
    if (!sandboxCode.trim()) return;

    const newFile: JavaFile = {
      name: sandboxFileName.endsWith(".java") ? sandboxFileName : `${sandboxFileName}.java`,
      path: `sandbox/${sandboxFileName.endsWith(".java") ? sandboxFileName : `${sandboxFileName}.java`}`,
      content: sandboxCode,
      originalContent: sandboxCode
    };

    setLoadedFiles([newFile, ...loadedFiles]);
    setActiveFileIndex(0);
    setSandboxCode("");
    setIsSandboxOpen(false);
    setScanState("idle");
    setVulnerabilities([]);
    setSelectedVulnerabilityId(null);
    setDroppedItemName(newFile.name);
  };

  // Perform full Code Security Scan
  const triggerCodeScan = async () => {
    if (loadedFiles.length === 0) return;

    setScanState("scanning");
    setIsSingleScanActive(false);
    setScanProgress(5);
    setScanProgressText("Initializing security context...");

    // Dramatic progress bar steps
    const steps = [
      { prg: 20, text: "Generating AST representation models..." },
      { prg: 45, text: "Tracing data tainted parameters..." },
      { prg: 70, text: "Cross-referencing OWASP rulesets..." },
      { prg: 88, text: "Verifying cryptographic strength limits..." },
      { prg: 95, text: "Assembling vulnerability matrix maps..." }
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setScanProgress(steps[currentStep].prg);
        setScanProgressText(steps[currentStep].text);
        currentStep++;
      }
    }, 550);

    try {
      const response = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: loadedFiles })
      });

      clearInterval(interval);
      setScanProgress(100);
      setScanProgressText("Audit compilation finalized!");

      const data = await response.json();

      setTimeout(() => {
        const uniqueVulns = new Map<string, Vulnerability>();
        (data.vulnerabilities || []).forEach((v: Vulnerability, idx: number) => {
          const sig = `${v.filePath}:${v.lineNumber}:${v.type}:${v.cwe_id}`;
          if (!uniqueVulns.has(sig)) {
            uniqueVulns.set(sig, {
              ...v,
              id: v.id || `vuln-${Date.now()}-${idx}-${Math.random().toString(36).substring(7)}`
            });
          }
        });
        const vulnsWithIds = Array.from(uniqueVulns.values());
        setVulnerabilities(vulnsWithIds);

        // Setup state map of vulnerabilities
        const stateMap: Record<string, "idle" | "fixing" | "diff_ready" | "approved"> = {};
        vulnsWithIds.forEach((v: Vulnerability) => {
          stateMap[v.id] = v.remediatedSnippet ? "diff_ready" : "idle";
        });
        setRemediatingIdState(stateMap);

        setScanMode(data.mode || "local_heuristics");
        setScanMessage(data.message || "");
        setScanState("completed");

        // Save to History
        setScanHistory(prev => [{
          id: Date.now().toString(),
          timestamp: new Date(),
          mode: data.mode || "local_heuristics",
          files: [...loadedFiles],
          vulnerabilities: vulnsWithIds
        }, ...prev]);
      }, 300);

    } catch (err: any) {
      clearInterval(interval);
      console.error("Scan attempt failed completely:", err);
      setScanProgressText("API Interface failed. Starting local scanner...");

      // Secondary fallback
      setTimeout(() => {
        setScanState("idle");
        alert("Server communication error. Please ensure the backend dev server is active and running.");
      }, 1000);
    }
  };

  // Perform Single File Scan
  const triggerSingleFileScan = async () => {
    if (!activeFile) return;

    setScanState("scanning");
    setIsSingleScanActive(true);
    setScanProgress(5);
    setScanProgressText(`Initializing scan for ${activeFile.name}...`);

    const steps = [
      { prg: 20, text: "Generating AST representation models..." },
      { prg: 45, text: "Tracing data tainted parameters..." },
      { prg: 70, text: "Cross-referencing OWASP rulesets..." },
      { prg: 88, text: "Verifying cryptographic strength limits..." },
      { prg: 95, text: "Assembling vulnerability matrix maps..." }
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setScanProgress(steps[currentStep].prg);
        setScanProgressText(steps[currentStep].text);
        currentStep++;
      }
    }, 550);

    try {
      const response = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: [activeFile] })
      });

      clearInterval(interval);
      setScanProgress(100);
      setScanProgressText("Audit compilation finalized!");

      const data = await response.json();

      setTimeout(() => {
        const uniqueVulns = new Map<string, Vulnerability>();
        (data.vulnerabilities || []).forEach((v: Vulnerability, idx: number) => {
          const sig = `${v.filePath}:${v.lineNumber}:${v.type}:${v.cwe_id}`;
          if (!uniqueVulns.has(sig)) {
            uniqueVulns.set(sig, {
              ...v,
              id: v.id || `vuln-${Date.now()}-${idx}-${Math.random().toString(36).substring(7)}`
            });
          }
        });
        const vulnsWithIds = Array.from(uniqueVulns.values());
        setVulnerabilities(vulnsWithIds);

        const stateMap: Record<string, "idle" | "fixing" | "diff_ready" | "approved"> = {};
        vulnsWithIds.forEach((v: Vulnerability) => {
          stateMap[v.id] = v.remediatedSnippet ? "diff_ready" : "idle";
        });
        setRemediatingIdState(stateMap);

        setScanMode(data.mode || "local_heuristics");
        setScanMessage(data.message || "");
        setScanState("completed");

        setScanHistory(prev => [{
          id: Date.now().toString(),
          timestamp: new Date(),
          mode: data.mode || "local_heuristics",
          files: [activeFile],
          vulnerabilities: vulnsWithIds
        }, ...prev]);
      }, 300);

    } catch (err: any) {
      clearInterval(interval);
      console.error("Scan attempt failed completely:", err);
      setScanProgressText("API Interface failed.");

      setTimeout(() => {
        setScanState("idle");
        alert("Server communication error. Please ensure the backend dev server is active and running.");
      }, 1000);
    }
  };

  // Trigger Remediation Workflow ("Fix" Action)
  const triggerRemediation = async (vuln: Vulnerability) => {
    const fileToFix = loadedFiles.find(f => f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop());
    if (!fileToFix) {
      alert("Associated file contexts could not be matched. Make sure files are loaded.");
      return;
    }

    setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "fixing" }));

    try {
      const response = await fetch("/api/remediate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filePath: fileToFix.path,
          fileContent: fileToFix.content,
          vulnerability: vuln
        })
      });

      const data = await response.json();

      setVulnerabilities(prev => prev.map(v => {
        if (v.id === vuln.id) {
          return {
            ...v,
            remediatedSnippet: data.remediatedSnippet,
            remediationExplanation: data.remediationExplanation,
            // also keep track of what the full repaired file will look like
            fullRemediatedContent: data.fullRemediatedContent,
            snippet: v.snippet || fileToFix.content
          } as any;
        }
        return v;
      }));

      setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "diff_ready" }));

    } catch (err) {
      console.error("Remediation repair API trigger failed:", err);
      alert("Error generating fix. Please retry or adjust code.");
      setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "idle" }));
    }
  };

  // Action: Ignore Card
  const handleIgnoreVulnerability = (id: string) => {
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === id) {
        return { ...v, status: "Ignored" };
      }
      return v;
    }));
  };

  // Action: Not a Vulnerability
  const handleRemoveVulnerability = (id: string) => {
    setVulnerabilities(prev => prev.filter(v => v.id !== id));
    if (selectedVulnerabilityId === id) {
      setSelectedVulnerabilityId(null);
    }
  };

  // Action: Approve Remediation
  const handleApproveRemediation = (vuln: any) => {
    // 1. Permanently update original file's state
    if (vuln.fullRemediatedContent) {
      setLoadedFiles(prev => prev.map(f => {
        if (f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop()) {
          return {
            ...f,
            content: vuln.fullRemediatedContent
          };
        }
        return f;
      }));
    }

    // 2. Mark card as approved
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === vuln.id) {
        return { ...v, status: "Approved" };
      }
      return v;
    }));

    setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "approved" }));
  };

  // Action: Reject Remediation
  const handleRejectRemediation = (id: string) => {
    setRemediatingIdState(prev => ({ ...prev, [id]: "idle" }));
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === id) {
        return {
          ...v,
          remediatedSnippet: undefined,
          remediationExplanation: undefined
        } as any;
      }
      return v;
    }));
  };

  // Count summary stats
  const stats = useMemo(() => {
    const totalCount = vulnerabilities.length;
    const active = vulnerabilities.filter(v => v.status !== "Ignored" && v.status !== "Approved");
    const high = active.filter(v => v.severity === "High").length;
    const medium = active.filter(v => v.severity === "Medium").length;
    const low = active.filter(v => v.severity === "Low").length;
    const fixed = vulnerabilities.filter(v => v.status === "Approved").length;

    return {
      total: totalCount,
      activeCount: active.length,
      high,
      medium,
      low,
      fixed
    };
  }, [vulnerabilities]);

  // Click a vulnerability to highlight that line in IDE workspace
  const handleVulnerabilityClick = (v: Vulnerability) => {
    setSelectedVulnerabilityId(v.id);

    // Find matching active file index and transition IDE focus
    const matchedIdx = loadedFiles.findIndex(f => f.path === v.filePath || f.name === v.filePath.split('/').pop());
    if (matchedIdx !== -1) {
      setActiveFileIndex(matchedIdx);

      // Auto scroll code editor container to lines of interest, if possible
      setTimeout(() => {
        const targetElement = document.getElementById(`line-anchor-${v.lineNumber}`);
        if (targetElement) {
          targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
    }
  };

  // Action: Clear entire workspace
  const handleClearWorkspace = () => {
    setLoadedFiles([]);
    setActiveFileIndex(0);
    setVulnerabilities([]);
    setScanState("idle");
    setScanProgress(0);
    setScanProgressText("");
    setSelectedVulnerabilityId(null);
    setDroppedItemName(null);
    setScanMode("none");
    setScanMessage("");
    setIsSingleScanActive(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-[#fbcfe8]/35 via-[#e6f4ea]/45 to-[#ccfbf1]/40 text-slate-800 font-sans p-4 md:p-6 selection:bg-emerald-500/20 selection:text-emerald-950 relative overflow-hidden">
      {/* Background colorful glassmorphic blur shapes with hints of green */}
      <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] rounded-full bg-gradient-to-tr from-pink-300/30 to-emerald-255/20 blur-[140px] opacity-70 pointer-events-none animate-pulse duration-[10s]" />
      <div className="absolute bottom-[5%] right-[-10%] w-[55vw] h-[55vw] rounded-full bg-gradient-to-tr from-sky-200/25 via-[#e6f4ea]/30 to-emerald-300/25 blur-[160px] opacity-75 pointer-events-none animate-pulse duration-[15s]" />
      <div className="absolute top-[25%] right-[10%] w-[45vw] h-[45vw] rounded-full bg-gradient-to-tr from-emerald-300/25 via-teal-200/20 to-green-300/25 blur-[140px] opacity-75 pointer-events-none animate-pulse duration-[12s]" />

      {/* Container holding general borders */}
      <div className="max-w-[1550px] mx-auto space-y-6 relative z-10">

        {/* HEADER SECTION */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center border border-white/60 bg-white/60 p-5 rounded-3xl gap-4 backdrop-blur-xl shadow-xl shadow-slate-200/30">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-emerald-500 via-teal-500 to-indigo-600 text-white p-3.5 rounded-2xl shadow-lg shadow-emerald-500/20 animate-bounce-slow">
              <Shield className="w-7 h-7" />
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tight bg-gradient-to-r from-rose-600 via-indigo-600 to-emerald-600 bg-clip-text text-transparent flex items-center gap-2">
                CodeElixir.AI
                <span className="text-[10px] bg-emerald-500/10 text-emerald-700 border border-emerald-500/20 px-2.5 py-0.5 rounded-full uppercase tracking-widest font-mono font-bold">
                  JAVA AUDITOR v1.1
                </span>
              </h1>
              <p className="text-xs text-slate-600 mt-1 font-medium leading-relaxed">
                Execute deep AST-level data flow threat scanning and secure prompt remediation using server-contained LLM engines.
              </p>
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full md:w-auto">
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Loaded Files</div>
              <div className="text-lg font-extrabold text-slate-800 mt-0.5">{loadedFiles.length}</div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Threats Active</div>
              <div className="text-lg font-extrabold text-rose-600 mt-0.5 flex justify-center items-center gap-1.5">
                {stats.activeCount}
                {stats.high > 0 && <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />}
              </div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Severity Matrix</div>
              <div className="text-xs text-slate-700 font-mono mt-1 flex justify-center gap-1">
                <span className="text-rose-700 bg-rose-105/50 px-1 rounded border border-rose-200 text-[10px]" title="High">{stats.high}H</span>
                <span className="text-amber-700 bg-amber-105/50 px-1 rounded border border-amber-200 text-[10px]" title="Medium">{stats.medium}M</span>
              </div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm hover:border-emerald-300/60 transition-colors">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Auto Patches</div>
              <div className="text-lg font-extrabold text-emerald-600 mt-0.5 flex justify-center items-center gap-1">
                {stats.fixed} <Check className="w-4 h-4 text-emerald-500 stroke-[3]" />
              </div>
            </div>
          </div>
        </header>

        {/* MAIN NAVIGATION TABS */}
        <div className="flex gap-2 border-b border-slate-200 pb-px">
          <button
            onClick={() => setActiveTab("scan")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "scan"
                ? "bg-white/60 text-emerald-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <LayoutDashboard className="w-4 h-4" /> Scanner Workspace
          </button>
          <button
            onClick={() => setActiveTab("history")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "history"
                ? "bg-white/60 text-indigo-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <History className="w-4 h-4" /> Scan History
            {scanHistory.length > 0 && (
              <span className="bg-indigo-100 text-indigo-700 text-[10px] px-2 py-0.5 rounded-full">{scanHistory.length}</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("rocm")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "rocm"
                ? "bg-white/60 text-cyan-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <Cpu className="w-4 h-4" /> ROCm Stats
          </button>
        </div>

        {/* WORKSPACE DIVIDER GRID */}
        {activeTab === "scan" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

            {/* LEFT COLUMN: CONTROL & FINDINGS PANEL (7 COLS) */}
            <div className="lg:col-span-7 space-y-6">

              {/* CONTROL AND LOADING INTERFACES */}
              <section className="border border-white/60 bg-white/50 p-5 rounded-3xl space-y-4 backdrop-blur-md shadow-xl shadow-slate-200/20">
                <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                  <FolderOpen className="w-4 h-4 text-emerald-500" /> File & Project Selection
                </h2>



                {/* Advanced Drag & Drop / File selection area */}
                <div
                  className={`transition-all border-2 border-dashed rounded-2xl p-5 text-center cursor-pointer ${isDragging
                      ? "border-rose-450 bg-rose-500/5 text-rose-700"
                      : "border-slate-200/80 bg-white/40 hover:border-slate-350 hover:bg-white/70 text-slate-500"
                    }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="p-3 bg-rose-50 text-rose-500 rounded-2xl border border-rose-100">
                      <FolderOpen className="w-6 h-6" />
                    </div>
                    <p className="text-xs text-slate-700 font-bold">
                      {droppedItemName ? `Loaded: ${droppedItemName}` : "Drag and Drop .java files or folder directories here"}
                    </p>
                    <p className="text-[10px] text-slate-400 font-medium max-w-sm mx-auto">
                      Supports selecting multiple java classes or entire project folders.
                    </p>

                    <div className="flex gap-3 justify-center mt-2">
                      <button
                        onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                        className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-xs font-bold hover:bg-slate-50 hover:border-slate-300 text-slate-600 shadow-sm transition-all flex items-center gap-1.5"
                      >
                        Select Files
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }}
                        className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-xs font-bold hover:bg-slate-50 hover:border-slate-300 text-slate-600 shadow-sm transition-all flex items-center gap-1.5"
                      >
                        Select Folder
                      </button>
                    </div>
                  </div>

                  {/* Hidden File Inputs */}
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    multiple
                    accept=".java"
                    className="hidden"
                  />
                  <input
                    type="file"
                    ref={folderInputRef}
                    onChange={handleFileChange}
                    webkitdirectory="true"
                    className="hidden"
                  />
                </div>

                {/* Selection Status Banner */}
                <div className="flex flex-wrap justify-between items-center bg-slate-50 p-3 rounded-2xl border border-slate-200/60 gap-3">
                  <div className="text-xs text-slate-600 flex items-center gap-2 font-medium">
                    <FileCode className="w-4 h-4 text-indigo-500" />
                    Status: <span className="font-mono text-slate-800 font-bold">{loadedFiles.length} java source paths parsed.</span>
                  </div>

                  {loadedFiles.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 mt-2 sm:mt-0">
                      <button
                        onClick={handleClearWorkspace}
                        disabled={scanState === "scanning"}
                        className={`px-4 py-2.5 rounded-xl text-xs font-bold shadow-sm transition-all flex items-center gap-2 ${scanState === "scanning"
                            ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
                            : "bg-white text-slate-600 hover:text-rose-600 hover:bg-rose-50 border border-slate-200"
                          }`}
                        title="Clear Workspace"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> Clear Workspace
                      </button>
                      <button
                        onClick={triggerCodeScan}
                        disabled={scanState === "scanning"}
                        className={`px-5 py-2.5 rounded-xl text-xs font-bold tracking-wide shadow-md transition-all flex items-center gap-2 ${scanState === "scanning"
                            ? "bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300"
                            : "bg-gradient-to-r from-rose-500 to-indigo-600 text-white hover:opacity-90 active:scale-95 cursor-pointer"
                          }`}
                      >
                        {scanState === "scanning" ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin text-rose-450" /> Evaluating AST...
                          </>
                        ) : (
                          <>
                            <Cpu className="w-3.5 h-3.5" /> Analyze Java Project
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>

                {/* PROGRESS VIEW FOR ACTIVE SCANNING */}
                <AnimatePresence>
                  {scanState === "scanning" && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="bg-white/80 p-4 border border-slate-250 rounded-2xl space-y-2 shadow-sm"
                    >
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-rose-600 flex items-center gap-2 font-mono font-bold">
                          <Terminal className="w-3 h-3 text-rose-500 animate-pulse" /> {scanProgressText}
                        </span>
                        <span className="font-mono font-bold text-slate-700">{scanProgress}%</span>
                      </div>

                      {/* Visual Progress Bar Wrapper */}
                      <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200/60">
                        <motion.div
                          className="bg-gradient-to-r from-rose-500 via-pink-400 to-indigo-500 h-full rounded-full"
                          initial={{ width: "0%" }}
                          animate={{ width: `${scanProgress}%` }}
                          transition={{ duration: 0.1 }}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </section>

              {/* PASTE DIALOG SHEETS */}
              <AnimatePresence>
                {isSandboxOpen && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-white/95 backdrop-blur-xl border border-white/60 p-5 rounded-3xl space-y-4 shadow-2xl shadow-indigo-100"
                  >
                    <div className="flex justify-between items-center border-b border-slate-100 pb-2">
                      <h3 className="text-xs font-bold font-mono text-slate-800 flex items-center gap-2">
                        <Code2 className="w-4 h-4 text-rose-500" /> Java Coding SandBox Playground
                      </h3>
                      <button
                        onClick={() => setIsSandboxOpen(false)}
                        className="text-slate-400 hover:text-slate-600"
                      >
                        <XCircle className="w-5 h-5" />
                      </button>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <label className="text-[10px] font-mono text-slate-400 block mb-1">Dummy Filename (.java)</label>
                        <input
                          type="text"
                          value={sandboxFileName}
                          onChange={(e) => setSandboxFileName(e.target.value)}
                          className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs text-slate-800 font-mono focus:border-rose-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-slate-400 block mb-1">Java Source Code</label>
                        <textarea
                          rows={12}
                          value={sandboxCode}
                          onChange={(e) => setSandboxCode(e.target.value)}
                          className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-800 font-mono focus:border-rose-500 focus:outline-none focus:ring-1 focus:ring-rose-500/30"
                          placeholder="Paste your custom vulnerable Class structures..."
                        />
                      </div>
                    </div>
                    <div className="flex justify-end gap-2 pt-2">
                      <button
                        onClick={() => setIsSandboxOpen(false)}
                        className="px-3.5 py-1.5 text-xs text-slate-400 hover:text-slate-600 transition-all font-semibold"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleAddSandboxFile}
                        className="px-4 py-2 text-xs bg-rose-500 hover:bg-rose-600 transition-all text-white font-bold rounded-xl shadow-md shadow-rose-500/10"
                      >
                        Load Into Explorer
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* VULNERABILITY CARDS SECTION */}
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                    <Flame className="w-4 h-4 text-rose-500" /> Scan Results Findings ({vulnerabilities.filter(v => v.status !== 'Ignored').length})
                  </h2>

                  {/* Active scan engines declaration */}
                  {scanState === "completed" && (
                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${scanMode === "gemini_ai"
                        ? "bg-purple-50 text-purple-600 border-purple-200 font-bold"
                        : "bg-slate-100 text-slate-600 border-slate-200"
                      }`}>
                      Engine: {scanMode === "gemini_ai" ? "Gemini AI (Complete AST API)" : "Offline Static Heuristics Backend"}
                    </span>
                  )}
                </div>

                {scanState === "idle" && (
                  <div className="border border-white/65 bg-white/45 p-10 rounded-3xl text-center space-y-4 shadow-sm backdrop-blur-md">
                    <div className="mx-auto w-12 h-12 bg-white border border-slate-200 rounded-2xl flex items-center justify-center text-slate-400 shadow-sm">
                      <Shield className="w-6 h-6 text-indigo-500" />
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-bold text-slate-800">No active audit logs available</p>
                      <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed font-semibold">
                        Review files or select a preset project, and hit <strong className="text-rose-500 font-bold">"Analyze Java Project"</strong> above to extract findings.
                      </p>
                    </div>
                  </div>
                )}

                {scanState === "completed" && vulnerabilities.length === 0 && (
                  <div className="border border-white/65 bg-white/45 p-10 rounded-3xl text-center space-y-2 shadow-sm">
                    <CheckCircle2 className="w-10 h-10 text-emerald-500 mx-auto" />
                    <p className="text-sm font-bold text-slate-800">Clean Bill of Health!</p>
                    <p className="text-xs text-slate-500 font-semibold">We could not identify any vulnerability parameters inside scanned Java files.</p>
                  </div>
                )}

                {scanState === "completed" && (
                  <div className="space-y-4">
                    {vulnerabilities.map((v) => {
                      const localState = remediatingIdState[v.id] || "idle";

                      // Hide Card if Ignored
                      if (v.status === "Ignored") return null;

                      const isHigh = v.severity === "High";
                      const isMed = v.severity === "Medium";
                      const isLow = v.severity === "Low";
                      const isApproved = v.status === "Approved";

                      return (
                        <motion.article
                          key={v.id}
                          layoutId={`card-${v.id}`}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`border rounded-3xl overflow-hidden transition-all duration-300 bg-white/60 backdrop-blur-md relative ${selectedVulnerabilityId === v.id
                              ? "border-rose-500 ring-2 ring-rose-500/15 shadow-lg"
                              : isApproved
                                ? "border-emerald-200 opacity-80"
                                : "border-white/70 hover:border-slate-300 shadow-sm"
                            }`}
                          onClick={() => handleVulnerabilityClick(v)}
                        >
                          {/* High priority attention bar */}
                          {/* File Path Header Over the Card */}
                          <div className="bg-slate-800 text-slate-200 text-[10px] font-mono px-4 py-1.5 font-bold truncate w-full flex items-center gap-2">
                            <FileCode className="w-3.5 h-3.5 text-emerald-400" />
                            {v.filePath}
                          </div>
                          {/* High priority attention bar */}
                          <div className={`h-1.5 w-full ${isApproved
                              ? "bg-emerald-500"
                              : isHigh
                                ? "bg-rose-500"
                                : isMed
                                  ? "bg-amber-500"
                                  : "bg-sky-400"
                            }`} />

                          <div className="p-5 space-y-4">

                            {/* Top Tag Header */}
                            <div className="flex flex-wrap justify-between items-center gap-2">
                              <div className="flex items-center gap-2 flex-wrap">
                                {/* Severity Badge */}
                                <span className={`text-[10px] uppercase tracking-wider font-bold px-2.5 py-0.5 rounded-full ${isApproved
                                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                                    : isHigh
                                      ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                                      : isMed
                                        ? "bg-amber-50 text-amber-700 border border-amber-200/60"
                                        : "bg-sky-50 text-sky-700 border border-sky-200/60"
                                  }`}>
                                  {isApproved ? "Resolved" : `${v.severity} Severity`}
                                </span>

                                <span className="text-xs font-bold text-slate-800 font-sans">
                                  {v.cwe_name || v.type}
                                </span>

                                {v.cwe_id && (
                                  <span className="text-[10px] bg-slate-100 border border-slate-200 px-2 py-0.5 rounded text-slate-600 font-mono">
                                    {v.cwe_id}
                                  </span>
                                )}
                                {v.confidence && (
                                  <span className="text-[10px] bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded text-indigo-600 font-mono">
                                    Conf: {(v.confidence * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>

                              {/* Line Number indicator */}
                              <span className="font-mono text-[9px] text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200/80 font-bold col-span-2">
                                Line: {v.location?.line || v.lineNumber}
                              </span>
                            </div>

                            {/* Vulnerability Description exploration */}
                            <div className="space-y-2">
                              <p className="text-xs text-slate-600 leading-relaxed font-semibold">
                                {v.description}
                              </p>
                              {v.impact && (
                                <p className="text-[11px] text-rose-700/80 leading-relaxed font-medium bg-rose-50 p-2 rounded-lg border border-rose-100">
                                  <strong>Impact:</strong> {v.impact}
                                </p>
                              )}
                              {v.recommendation && (
                                <p className="text-[11px] text-emerald-700/80 leading-relaxed font-medium bg-emerald-50 p-2 rounded-lg border border-emerald-100">
                                  <strong>Recommendation:</strong> {v.recommendation}
                                </p>
                              )}
                            </div>

                            {/* ACTION TOOLBAR */}
                            <div className="flex flex-wrap justify-between items-center border-t border-slate-100 pt-4 gap-3">

                              {/* Action Buttons for active logs */}
                              {!isApproved && localState === "idle" && (
                                <div className="flex gap-2">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      triggerRemediation(v);
                                    }}
                                    className="px-3.5 py-2 bg-gradient-to-r from-rose-500 via-rose-500 to-indigo-600 text-white font-bold text-xs rounded-xl transition-all shadow-md shadow-rose-500/10 cursor-pointer flex items-center gap-1.5"
                                  >
                                    <Sparkles className="w-3.5 h-3.5 animate-pulse" /> Fix Security Issue
                                  </button>

                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleIgnoreVulnerability(v.id);
                                    }}
                                    className="px-3 py-2 bg-white border border-slate-200 hover:text-amber-600 hover:border-amber-300 hover:bg-amber-50/10 text-slate-500 text-xs rounded-xl transition-all font-bold shadow-sm"
                                  >
                                    Ignore
                                  </button>

                                </div>
                              )}

                              {/* IF REMEDIATING LOADING STATE */}
                              {localState === "fixing" && (
                                <div className="text-xs text-rose-600 flex items-center gap-2 font-mono py-1 font-bold">
                                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-rose-550" />
                                  Triggering server model remediation repair mechanisms...
                                </div>
                              )}

                              {/* DIFF VIEW READY WORKFLOWS */}
                              <AnimatePresence>
                                {localState === "diff_ready" && v.remediatedSnippet && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="w-full space-y-4 pt-1"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    {/* Code Diff Display Row */}
                                    <div className="bg-white p-4 rounded-2xl border border-rose-200/50 space-y-3 shadow-inner">
                                      <h4 className="text-[9px] font-bold font-mono tracking-widest text-rose-700 uppercase flex items-center gap-2">
                                        <GitCommit className="w-3.5 h-3.5 text-rose-500" /> side-by-side remediation comparison
                                      </h4>

                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        {/* Old segment block */}
                                        <div className="bg-red-50/70 p-3 rounded-xl border border-red-100 text-left">
                                          <div className="text-red-750 font-sans text-[9px] mb-2 uppercase font-extrabold text-red-650">ORIGINAL</div>
                                          <div className="font-mono text-xs text-red-900 border-l-2 border-red-400 pl-2 select-all overflow-x-auto whitespace-pre max-w-full font-semibold">
                                            {v.snippet}
                                          </div>
                                        </div>

                                        {/* Corrected segment block */}
                                        <div className="bg-emerald-50/80 p-3 rounded-xl border border-emerald-100 text-left">
                                          <div className="text-emerald-700 font-sans text-[9px] mb-2 uppercase font-extrabold flex items-center gap-1.5">
                                            PROPOSED SECURE DEVIATION
                                            <Sparkles className="w-2.5 h-2.5 text-emerald-650 animated-pulse" />
                                          </div>
                                          <div className="font-mono text-xs text-emerald-800 border-l-2 border-emerald-450 pl-2 select-all overflow-x-auto whitespace-pre max-w-full font-bold">
                                            {v.remediatedSnippet}
                                          </div>
                                        </div>
                                      </div>

                                      {/* Detailed technical explanation */}
                                      {v.remediationExplanation && (
                                        <div className="mt-3 text-xs bg-slate-50 border border-slate-200 p-3 rounded-xl text-slate-600 flex gap-2">
                                          <Info className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
                                          <div className="space-y-1">
                                            <strong className="text-slate-800 text-[10px] uppercase tracking-wider block font-mono font-bold">Remediation Action</strong>
                                            <p className="leading-relaxed font-semibold text-slate-600">{v.remediationExplanation}</p>
                                          </div>
                                        </div>
                                      )}

                                      {/* Action approval triggers */}
                                      <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
                                        <button
                                          onClick={() => handleRejectRemediation(v.id)}
                                          className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-700 hover:bg-slate-50 rounded-xl transition-all font-bold"
                                        >
                                          Reject Corrective
                                        </button>

                                        <button
                                          onClick={() => handleApproveRemediation(v)}
                                          className="px-4 py-1.5 bg-emerald-500 hover:bg-emerald-600 border border-emerald-450 text-white font-extrabold text-xs rounded-xl transition-all shadow-md shadow-emerald-500/10 flex items-center gap-1.5 cursor-pointer"
                                        >
                                          <Check className="w-3.5 h-3.5" /> Approve & Apply Fix
                                        </button>
                                      </div>
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>

                              {/* APPROVED SUCCESS BANNER */}
                              {localState === "approved" && (
                                <motion.div
                                  initial={{ opacity: 0, scale: 0.95 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  className="w-full bg-emerald-50 text-emerald-800 border border-emerald-250 p-3.5 rounded-2xl flex items-center justify-between gap-2 shadow-sm"
                                >
                                  <div className="flex items-center gap-2 text-xs font-mono font-bold">
                                    <FileCheck className="w-4 h-4 text-emerald-650" />
                                    Vulnerability permanently resolved! File records corrected in active scope memory.
                                  </div>
                                  <span className="text-[9px] font-bold text-emerald-700 bg-emerald-200/50 uppercase border border-emerald-350 px-2.5 py-0.5 rounded-full font-sans">
                                    Approved & Committed
                                  </span>
                                </motion.div>
                              )}

                            </div>

                          </div>
                        </motion.article>
                      );
                    })}
                  </div>
                )}
              </div>

            </div>

            {/* RIGHT COLUMN: ACTIVE INTERACTIVE IDE / CODE VIEWER (5 COLS) */}
            <section className="lg:col-span-5 border border-white/60 bg-white/55 p-5 rounded-3xl flex flex-col h-[calc(100vh-140px)] min-h-[550px] sticky top-6 backdrop-blur-md shadow-xl shadow-slate-200/20">

              {/* Tab selection panel */}
              <div className="flex justify-between items-center border-b border-slate-200 pb-3 mb-4 shrink-0">
                <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-emerald-500" /> Workspace Native Files
                </h2>

                <span className="text-[9px] font-mono text-slate-500 bg-slate-50 border border-slate-250/80 px-2.5 py-1 rounded-lg font-bold">
                  {loadedFiles.length} File Records
                </span>
              </div>

              {/* Flat Explorer tab labels */}
              <div className="flex gap-1 overflow-x-auto pb-2 shrink-0 max-w-full">
                {loadedFiles.map((file, idx) => {
                  const isActive = activeFileIndex === idx;

                  // Count active vulnerabilities inside this specific file path to show a badge count
                  const fileVulns = vulnerabilities.filter(v => v.status !== "Ignored" && v.status !== "Approved" && (v.filePath === file.path || v.filePath === file.name));

                  return (
                    <button
                      key={file.path || file.name}
                      onClick={() => {
                        setActiveFileIndex(idx);
                        setSelectedVulnerabilityId(null);
                      }}
                      className={`px-3 py-2 text-xs font-mono rounded-xl transition-all shrink-0 border flex items-center gap-2 ${isActive
                          ? "bg-white text-slate-800 border-slate-300 shadow-md shadow-slate-200/40 font-bold"
                          : "bg-white/40 text-slate-500 border-slate-200/60 hover:bg-white/80 hover:text-slate-850"
                        }`}
                    >
                      <FileCode className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                      <span className="truncate max-w-[120px] font-sans">{file.name}</span>

                      {/* Vuln Indicators badge */}
                      {fileVulns.length > 0 && (
                        <span className="w-4.5 h-4.5 bg-rose-500 text-white text-[9px] font-bold rounded-full flex items-center justify-center shrink-0 shadow-sm">
                          {fileVulns.length}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Core Code Viewer Screen */}
              <div className="flex-1 min-h-0 bg-slate-900 rounded-2xl border border-slate-950 overflow-hidden flex flex-col relative shadow-xl">

                {activeFile ? (
                  <>
                    {/* Pseudo IDE status header */}
                    <div className="bg-slate-950 px-4 py-2 text-xs border-b border-slate-900/80 flex justify-between items-center shrink-0">
                      <span className="font-mono text-slate-400 truncate max-w-xs">{activeFile.path}</span>
                      <span className="font-mono text-[9px] text-slate-500 font-semibold">Parser: JAVA (UTF-8)</span>
                    </div>

                    {/* Lines container element */}
                    <div className="flex-1 overflow-y-auto font-mono text-xs p-4 leading-relaxed select-text select-all bg-slate-900/95 scrollbar-thin">
                      {activeFile.content.split("\n").map((lineContent, idx) => {
                        const lineNum = idx + 1;

                        // Check if this line corresponds to a critical active vulnerability
                        const matchingVuln = vulnerabilities.find(
                          v => v.status !== "Ignored" &&
                            v.status !== "Approved" &&
                            v.lineNumber === lineNum &&
                            (v.filePath === activeFile.path || v.filePath === activeFile.name)
                        );

                        const isLineMatched = matchingVuln !== undefined;
                        const isVulnerabilitySelected = matchingVuln && selectedVulnerabilityId === matchingVuln.id;

                        return (
                          <div
                            key={idx}
                            id={`line-anchor-${lineNum}`}
                            className={`flex items-start transition-all ${isVulnerabilitySelected
                                ? "bg-rose-500/20 border-l-4 border-rose-500 -ml-4 pl-3"
                                : isLineMatched
                                  ? "bg-rose-950/20 border-l-2 border-rose-450 -ml-4 pl-3.5 cursor-pointer text-rose-350 font-semibold"
                                  : "text-slate-300"
                              }`}
                            onClick={() => matchingVuln && handleVulnerabilityClick(matchingVuln)}
                            title={matchingVuln ? `Click to inspect: ${matchingVuln.type}` : undefined}
                          >
                            {/* Code line enumeration numbers */}
                            <div className="text-slate-500 w-7 select-none text-right pr-3 font-mono text-[10px] opacity-40 shrink-0">
                              {lineNum}
                            </div>

                            {/* Live content characters */}
                            <div className={`whitespace-pre break-all ${isLineMatched ? "font-bold bg-rose-500/10 px-1 rounded text-rose-100" : ""}`}>
                              {renderHighlightedCode(lineContent)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500 bg-slate-900">
                    <FileCode className="w-10 h-10 mb-2 opacity-30 text-emerald-500" />
                    <p className="text-xs">Select or upload Java file inputs to active explorer.</p>
                  </div>
                )}

              </div>

              {/* Interactive User documentation prompt block */}
              <div className="mt-4 shrink-0 flex gap-2">
                <button
                  onClick={triggerSingleFileScan}
                  disabled={!activeFile || scanState === "scanning"}
                  className={`w-full py-3 rounded-2xl text-xs font-bold tracking-wide shadow-md transition-all flex items-center justify-center gap-2 ${!activeFile || scanState === "scanning"
                      ? "bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300"
                      : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-emerald-500/20 active:scale-95 cursor-pointer"
                    }`}
                >
                  {scanState === "scanning" ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" /> Scanning active file...
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" /> Run Scan on Selected File
                    </>
                  )}
                </button>
              </div>

            </section>

          </div>
        )}

        {/* HISTORY TAB VIEW */}
        {activeTab === "history" && (
          <div className="border border-white/60 bg-white/50 p-6 rounded-3xl space-y-4 backdrop-blur-md shadow-xl shadow-slate-200/20">
            <h2 className="text-sm font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
              <RefreshCw className="w-5 h-5 text-indigo-500" /> Session Scan History
            </h2>

            {scanHistory.length === 0 ? (
              <div className="text-center p-12 text-slate-400 font-medium border border-dashed border-slate-300 rounded-2xl">
                No scans have been performed yet in this session.
              </div>
            ) : (
              <div className="space-y-6">
                {scanHistory.map((historyItem) => (
                  <div key={historyItem.id} className="bg-white/80 p-5 rounded-2xl border border-slate-200 shadow-sm relative group">
                    <button
                      onClick={() => setScanHistory(prev => prev.filter(h => h.id !== historyItem.id))}
                      className="absolute top-4 right-4 text-slate-300 hover:text-rose-500 transition-colors opacity-0 group-hover:opacity-100"
                      title="Remove from history"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>

                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
                          {historyItem.timestamp.toLocaleTimeString()}
                        </div>
                        <h3 className="font-bold text-slate-800">
                          {historyItem.files.length} File(s) Scanned
                        </h3>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-[10px] bg-slate-100 text-slate-600 border border-slate-200 px-2 py-1 rounded font-mono font-bold">
                          {historyItem.mode}
                        </span>
                        <span className={`text-[10px] px-2 py-1 rounded font-mono font-bold ${historyItem.vulnerabilities.length > 0 ? "bg-rose-100 text-rose-700 border border-rose-200" : "bg-emerald-100 text-emerald-700 border border-emerald-200"
                          }`}>
                          {historyItem.vulnerabilities.length} Threats Found
                        </span>
                      </div>
                    </div>

                    {historyItem.vulnerabilities.length > 0 && (
                      <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 space-y-3 mb-2">
                        <h4 className="text-xs font-bold text-slate-600 uppercase">Findings Details:</h4>
                        {Array.from(new Set(historyItem.vulnerabilities.map(v => v.filePath))).map((filePath, i) => {
                          const fileVulns = historyItem.vulnerabilities.filter(v => v.filePath === filePath);
                          const file = historyItem.files.find(f => f.path === filePath || f.name === filePath.split('/').pop());
                          const fileId = `${historyItem.id}-${filePath}`;
                          const isExpanded = expandedHistoryIds.includes(fileId);

                          return (
                            <div key={i} className="bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm">
                              <div className="p-3 flex justify-between items-start border-b border-slate-100 bg-slate-50/50">
                                <div>
                                  <div className="font-bold text-xs text-slate-700 flex items-center gap-2 mb-2">
                                    <FileCode className="w-3.5 h-3.5 text-slate-400" />
                                    {filePath}
                                  </div>
                                  <div className="space-y-1.5">
                                    {fileVulns.map((v, vIdx) => (
                                      <div key={vIdx} className="text-[11px] font-mono text-slate-600 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-rose-500 shrink-0" />
                                        <span className="font-bold text-rose-600">{v.type}</span> at line {v.lineNumber}
                                        {v.status === 'Approved' && <span className="text-[9px] bg-emerald-100 text-emerald-700 px-1.5 rounded-full py-0.5 font-bold uppercase">Resolved</span>}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                                <button
                                  onClick={() => toggleHistoryExpanded(fileId)}
                                  className="text-slate-400 hover:text-slate-700 transition-colors p-1.5 rounded-md hover:bg-slate-200 bg-white border border-slate-200 shadow-sm ml-2 mt-1 shrink-0"
                                  title={isExpanded ? "Collapse Source Code" : "Expand Source Code"}
                                >
                                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                </button>
                              </div>

                              {isExpanded && file && (
                                <div className="bg-[#0f172a] p-4 overflow-auto max-h-[400px]">
                                  <pre className="font-mono text-xs leading-relaxed text-slate-300">
                                    {file.content.split('\n').map((line, lineIdx) => (
                                      <div key={lineIdx} className="flex">
                                        <span className="w-8 shrink-0 text-slate-600 select-none text-right pr-3">{lineIdx + 1}</span>
                                        <span className="whitespace-pre break-all">
                                          {renderHighlightedCode(line)}
                                        </span>
                                      </div>
                                    ))}
                                  </pre>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ROCM GPU DIAGNOSTICS TAB */}
        {activeTab === "rocm" && (
          <div className="space-y-6">
            <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-6 shadow-xl shadow-slate-200/40 flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-indigo-50 border border-indigo-100 rounded-2xl shadow-sm">
                  <Cpu className="w-10 h-10 text-indigo-500 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-lg font-extrabold text-slate-800 flex items-center gap-2">
                    {rocmStats.gpuName}
                    <span className="text-xs bg-indigo-100 text-indigo-700 border border-indigo-200 font-bold px-2.5 py-0.5 rounded-full">
                      {rocmStats.gpuType}
                    </span>
                  </h2>
                  <p className="text-xs text-slate-500 mt-1 font-medium">
                    Hardware profiling stack mapping real-time processing nodes.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-4">
                <div className="bg-slate-50 border border-slate-200 p-4 rounded-2xl text-center min-w-[130px] shadow-sm">
                  <span className="text-[10px] font-bold text-slate-500 uppercase block">Models Loaded</span>
                  <div className="flex flex-col gap-1 mt-1">
                    {rocmStats.modelsLoaded.map((model: string, idx: number) => (
                      <span key={idx} className="text-xs font-extrabold text-purple-600 block font-mono bg-purple-50 px-2 py-0.5 rounded-md border border-purple-100">
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="bg-slate-50 border border-slate-200 p-4 rounded-2xl text-center min-w-[130px] shadow-sm">
                  <span className="text-[10px] font-bold text-slate-500 uppercase block">API Health</span>
                  <span className="text-xs font-extrabold text-emerald-600 block font-mono mt-1 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-100">
                    {rocmStats.apiHealth}
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-purple-600 tracking-wide mb-2 w-full text-center">VRAM Allocation (MiB)</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#9333ea" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - Math.min(1, rocmStats.vram / rocmStats.vramTotal))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.vram}</span>
                    <span className="text-[10px] text-purple-600 font-bold uppercase tracking-widest mt-1">/ {rocmStats.vramTotal} MiB</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-blue-500 tracking-wide mb-2 w-full text-center">Compute Engine Load</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#3b82f6" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.load / 100))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.load}%</span>
                    <span className="text-[10px] text-blue-500 font-bold uppercase tracking-widest mt-1">Utilisation</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-emerald-500 tracking-wide mb-2 w-full text-center">CPU Core Load</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#10b981" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.cpu / 100))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.cpu}%</span>
                    <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest mt-1">Utilisation</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-indigo-500 tracking-wide mb-2 w-full text-center">System RAM (GiB)</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#6366f1" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.ram / 256))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.ram}</span>
                    <span className="text-[10px] text-indigo-500 font-bold uppercase tracking-widest mt-1">/ 256 GiB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
