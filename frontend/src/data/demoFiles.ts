import { ProjectDemo } from '../types';

export const DEMO_PROJECTS: ProjectDemo[] = [
  {
    name: "E-Commerce Gateway (Web)",
    description: "Multi-vuln Java spring-like web controller handling search, exports, and customer reviews.",
    files: [
      {
        name: "OrderExportController.java",
        path: "src/main/java/com/store/controller/OrderExportController.java",
        originalContent: `package com.store.controller;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.springframework.web.bind.annotation.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

@RestController
@RequestMapping("/api/orders")
public class OrderExportController {

    private String dbUrl = "jdbc:mysql://localhost:3306/shopdb";
    private String dbUser = "prod_admin";
    private String dbPass = "SuperSecretDbPassword2026!"; // Hardcoded Admin Secret

    @GetMapping("/search")
    public String searchProduct(@RequestParam("query") String query) {
        try {
            Connection conn = DriverManager.getConnection(dbUrl, dbUser, dbPass);
            Statement stmt = conn.createStatement();
            // HIGH RISK: Direct string concatenation in SQL Query
            String sql = "SELECT * FROM products WHERE name LIKE '%" + query + "%' AND visible = 1";
            ResultSet rs = stmt.executeQuery(sql);
            
            StringBuilder sb = new StringBuilder();
            while (rs.next()) {
                sb.append(rs.getString("name")).append(", ");
            }
            return "Found: " + sb.toString();
        } catch (Exception e) {
            return "Error searching DB: " + e.getMessage();
        }
    }

    @PostMapping("/export")
    public String exportPdf(@RequestParam("format") String format, @RequestParam("reportId") String id) {
        try {
            // HIGH RISK: Command injection via shell-formatting parameter
            String command = "sh /opt/bin/generate_pdf.sh --report=" + id + " --format=" + format;
            Process process = Runtime.getRuntime().exec(command);
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\\n");
            }
            int exitCode = process.waitFor();
            return "Exit code " + exitCode + ". Output: " + output.toString();
        } catch (Exception e) {
            return "System Error during execution: " + e.getMessage();
        }
    }
}`,
        content: `package com.store.controller;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.springframework.web.bind.annotation.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

@RestController
@RequestMapping("/api/orders")
public class OrderExportController {

    private String dbUrl = "jdbc:mysql://localhost:3306/shopdb";
    private String dbUser = "prod_admin";
    private String dbPass = "SuperSecretDbPassword2026!"; // Hardcoded Admin Secret

    @GetMapping("/search")
    public String searchProduct(@RequestParam("query") String query) {
        try {
            Connection conn = DriverManager.getConnection(dbUrl, dbUser, dbPass);
            Statement stmt = conn.createStatement();
            // HIGH RISK: Direct string concatenation in SQL Query
            String sql = "SELECT * FROM products WHERE name LIKE '%" + query + "%' AND visible = 1";
            ResultSet rs = stmt.executeQuery(sql);
            
            StringBuilder sb = new StringBuilder();
            while (rs.next()) {
                sb.append(rs.getString("name")).append(", ");
            }
            return "Found: " + sb.toString();
        } catch (Exception e) {
            return "Error searching DB: " + e.getMessage();
        }
    }

    @PostMapping("/export")
    public String exportPdf(@RequestParam("format") String format, @RequestParam("reportId") String id) {
        try {
            // HIGH RISK: Command injection via shell-formatting parameter
            String command = "sh /opt/bin/generate_pdf.sh --report=" + id + " --format=" + format;
            Process process = Runtime.getRuntime().exec(command);
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\\n");
            }
            int exitCode = process.waitFor();
            return "Exit code " + exitCode + ". Output: " + output.toString();
        } catch (Exception e) {
            return "System Error during execution: " + e.getMessage();
        }
    }
}`
      },
      {
        name: "UserLoginService.java",
        path: "src/main/java/com/store/service/UserLoginService.java",
        originalContent: `package com.store.service;

import java.security.MessageDigest;
import java.util.Base64;

public class UserLoginService {

    public String hashPassword(String password) {
        try {
            // HIGH RISK: Vulnerable hashing algorithm (MD5) is cryptographically weak
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(password.getBytes("UTF-8"));
            
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            return null;
        }
    }

    public boolean checkCredentials(String username, String passInput, String storedHash) {
        // Compare stored hash with entered MD5 password
        String enteredHash = hashPassword(passInput);
        return enteredHash != null && enteredHash.equals(storedHash);
    }
}`,
        content: `package com.store.service;

import java.security.MessageDigest;
import java.util.Base64;

public class UserLoginService {

    public String hashPassword(String password) {
        try {
            // HIGH RISK: Vulnerable hashing algorithm (MD5) is cryptographically weak
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(password.getBytes("UTF-8"));
            
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            return null;
        }
    }

    public boolean checkCredentials(String username, String passInput, String storedHash) {
        // Compare stored hash with entered MD5 password
        String enteredHash = hashPassword(passInput);
        return enteredHash != null && enteredHash.equals(storedHash);
    }
}`
      }
    ]
  },
  {
    name: "Enterprise Admin & Integration Portal",
    description: "Enterprise service loading documents and loading external REST payloads securely.",
    files: [
      {
        name: "DocumentManagementService.java",
        path: "src/main/java/com/enterprise/integration/DocumentManagementService.java",
        originalContent: `package com.enterprise.integration;

import java.io.*;
import org.springframework.web.bind.annotation.*;
import javax.servlet.http.HttpServletResponse;

@RestController
@RequestMapping("/api/docs")
public class DocumentManagementService {

    private static final String BASE_DIR = "/var/shared/documents/";

    @GetMapping("/download")
    public void getDocument(@RequestParam("filename") String filename, HttpServletResponse response) {
        try {
            // HIGH RISK: Path Traversal vulnerability allows reading arbitrary files
            File file = new File(BASE_DIR + filename);
            if (!file.exists()) {
                response.sendError(404, "File not found");
                return;
            }
            
            response.setContentType("application/octet-stream");
            response.setHeader("Content-Disposition", "attachment; filename=\\"" + file.getName() + "\\"");
            
            FileInputStream fis = new FileInputStream(file);
            OutputStream os = response.getOutputStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            fis.close();
            os.flush();
        } catch (IOException e) {
            try {
                response.sendError(500, "Inward storage error");
            } catch (Exception ignored) {}
        }
    }
}`,
        content: `package com.enterprise.integration;

import java.io.*;
import org.springframework.web.bind.annotation.*;
import javax.servlet.http.HttpServletResponse;

@RestController
@RequestMapping("/api/docs")
public class DocumentManagementService {

    private static final String BASE_DIR = "/var/shared/documents/";

    @GetMapping("/download")
    public void getDocument(@RequestParam("filename") String filename, HttpServletResponse response) {
        try {
            // HIGH RISK: Path Traversal vulnerability allows reading arbitrary files
            File file = new File(BASE_DIR + filename);
            if (!file.exists()) {
                response.sendError(404, "File not found");
                return;
            }
            
            response.setContentType("application/octet-stream");
            response.setHeader("Content-Disposition", "attachment; filename=\\"" + file.getName() + "\\"");
            
            FileInputStream fis = new FileInputStream(file);
            OutputStream os = response.getOutputStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            fis.close();
            os.flush();
        } catch (IOException e) {
            try {
                response.sendError(500, "Inward storage error");
            } catch (Exception ignored) {}
        }
    }
}`
      },
      {
        name: "RemoteFetchService.java",
        path: "src/main/java/com/enterprise/integration/RemoteFetchService.java",
        originalContent: `package com.enterprise.integration;

import java.net.URL;
import java.net.HttpURLConnection;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/fetch")
public class RemoteFetchService {

    @PostMapping("/gadget")
    public String fetchWebhookData(@RequestParam("targetUrl") String targetUrl) {
        try {
            // HIGH RISK: Server-Side Request Forgery via unvalidated remote connection
            URL url = new URL(targetUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);
            
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            StringBuilder response = new StringBuilder();
            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            return response.toString();
        } catch (Exception e) {
            return "Error retrieving payload: " + e.getMessage();
        }
    }
}`,
        content: `package com.enterprise.integration;

import java.net.URL;
import java.net.HttpURLConnection;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/fetch")
public class RemoteFetchService {

    @PostMapping("/gadget")
    public String fetchWebhookData(@RequestParam("targetUrl") String targetUrl) {
        try {
            // HIGH RISK: Server-Side Request Forgery via unvalidated remote connection
            URL url = new URL(targetUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);
            
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            StringBuilder response = new StringBuilder();
            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            return response.toString();
        } catch (Exception e) {
            return "Error retrieving payload: " + e.getMessage();
        }
    }
}`
      }
    ]
  }
];

// Fallback high-fidelity scan results if Gemini API key is missing or for initial load
export const FALLBACK_VULNERABILITIES = [
  {
    id: "vuln-order-export-1",
    type: "SQL Injection",
    severity: "High" as const,
    filePath: "src/main/java/com/store/controller/OrderExportController.java",
    lineNumber: 24,
    snippet: 'String sql = "SELECT * FROM products WHERE name LIKE \'%" + query + "%\' AND visible = 1";',
    description: "The application concatenates untrusted user input directly into a SQL query string. This allows attackers to manipulate the database query, bypass authentication, retrieve, modify or delete database data, or execute administrative commands.",
    recommendation: "Use Parameterized Queries (PreparedStatement) instead of raw string concatenation. Bind parameters properly to ensure inputs are escaped correctly.",
    status: "Scanned" as const,
    remediatedSnippet: `String sql = "SELECT * FROM products WHERE name LIKE ? AND visible = 1";\nPreparedStatement pstmt = conn.prepareStatement(sql);\npst_stmt.setString(1, "%" + query + "%");\nResultSet rs = pstmt.executeQuery();`,
    remediationExplanation: "Replaced raw string concatenation with a Parameterized Query using `PreparedStatement`. The SQL query structure is pre-compiled, and the user-supplied query is treated strictly as a data parameter rather than executable SQL instruction."
  },
  {
    id: "vuln-order-export-2",
    type: "Command Injection",
    severity: "High" as const,
    filePath: "src/main/java/com/store/controller/OrderExportController.java",
    lineNumber: 41,
    snippet: 'String command = "sh /opt/bin/generate_pdf.sh --report=" + id + " --format=" + format;',
    description: "Using parameter values received immediately from API endpoints inside full shell commands without strict serialization or parameter separation allows command injection. Attackers can execute arbitrary command shell directives.",
    recommendation: "Avoid executing system commands directly if possible. If inevitable, separate arguments into a String array to execute directly via the operating system without spawning a shell command interpreter, or rigorously whitelist input parameters.",
    status: "Scanned" as const,
    remediatedSnippet: `String[] commandArgs = { "sh", "/opt/bin/generate_pdf.sh", "--report", id, "--format", format };\nProcess process = new ProcessBuilder(commandArgs).start();`,
    remediationExplanation: "Migrated process execution to use standard `ProcessBuilder` with array parameters. This forces the operating system to pass arguments safely directly to the executable, preventing shell injection operators like `;`, `&&`, or `|` from triggering."
  },
  {
    id: "vuln-user-login-1",
    type: "Cryptographic Weakness",
    severity: "Medium" as const,
    filePath: "src/main/java/com/store/service/UserLoginService.java",
    lineNumber: 11,
    snippet: 'MessageDigest md = MessageDigest.getInstance("MD5");',
    description: "MD5 is a cryptographically broken hashing algorithm prone to rapid hash collision generation. It must never be used to store or verify user passwords.",
    recommendation: "Upgrade hashing schemas to a strong password-hashing algorithm like bcrypt (using libraries like jBCrypt) or PBKDF2 with appropriate iterations.",
    status: "Scanned" as const,
    remediatedSnippet: `// Use Argon2 or bcrypt for proper, high-entropy password hashing\n// Example using PBKDF2WithHmacSHA256 (Built-in JRE mechanism) or jbcrypt:\n// String hashed = BCrypt.hashpw(password, BCrypt.gensalt(12));`,
    remediationExplanation: "Specified upgrading the low-entropy MD5 hash mechanism to BCrypt or PBKDF2WithHmacSHA256. This includes cryptographic salts and computation work factors to slow down rainbow table attacks."
  },
  {
    id: "vuln-doc-mgmt-1",
    type: "Path Traversal",
    severity: "High" as const,
    filePath: "src/main/java/com/enterprise/integration/DocumentManagementService.java",
    lineNumber: 17,
    snippet: 'File file = new File(BASE_DIR + filename);',
    description: "The application constructs a file path by joining a base folder with a client-supplied filename parameter. Directory traversal tokens (e.g. '../') in the parameter allow accessing arbitrary files outside the intended subfolder.",
    recommendation: "Validate that the canonical path of the resolved file starts exactly with the canonical path of the target base directory. Alternatively, whitelist filenames or map filenames to database IDs.",
    status: "Scanned" as const,
    remediatedSnippet: `File file = new File(BASE_DIR + filename);\nString baseCanonical = new File(BASE_DIR).getCanonicalPath();\nString fileCanonical = file.getCanonicalPath();\nif (!fileCanonical.startsWith(baseCanonical)) {\n    throw new SecurityException("Unauthorized directory access attempt detected");\n}`,
    remediationExplanation: "Added path canonicalization checking. Before reading the target file, we retrieve the dynamic canonical path recursively, confirming the resulting folder strictly matches the intended secure directory structure."
  },
  {
    id: "vuln-remote-fetch-1",
    type: "Server-Side Request Forgery",
    severity: "High" as const,
    filePath: "src/main/java/com/enterprise/integration/RemoteFetchService.java",
    lineNumber: 15,
    snippet: 'URL url = new URL(targetUrl);',
    description: "The server resolves and sends an HTTP requests to a target URL directly supplied by the API user. Attackers can trigger requests targeting local/internal services (e.g., localhost, 127.0.0.1, or cloud metadata IP 169.254.169.254).",
    recommendation: "Filter and restrict input URLs against a strict whitelist of approved domains. Validate target IP resolutions to block RFC 1918 private / loopback IP address ranges.",
    status: "Scanned" as const,
    remediatedSnippet: `URL url = new URL(targetUrl);\nString host = url.getHost();\n// Resolve host IP, ensure the target does NOT reside in private/loopback subnet range\nInetAddress ip = InetAddress.getByName(host);\nif (ip.isLoopbackAddress() || ip.isSiteLocalAddress()) {\n    throw new SecurityException("SSRF Attempt Target Blocked: Local or internal address space.");\n}\nHttpURLConnection conn = (HttpURLConnection) url.openConnection();`,
    remediationExplanation: "Resolved target DNS IPs to block internal loopback (127.0.0.1) and private RFC 1918 subnets (10.0.0.0/8, 192.168.0.0/16, etc.) from HTTP queries, mitigating loopback server probing."
  }
];
