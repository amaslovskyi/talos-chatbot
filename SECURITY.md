# Security Policy

## 🔒 Security Overview

This document outlines the security measures implemented in the RAG Chatbot project to protect sensitive data and ensure safe operation.

## 🛡️ Security Measures

### **🔐 Credential Management**
- **Environment Variables**: All API keys and secrets are managed through environment variables
- **No Hardcoded Secrets**: Zero hardcoded credentials in the codebase
- **Template Approach**: Setup scripts create placeholder values, users must provide real credentials
- **Secure Defaults**: Local models preferred over external APIs to minimize credential exposure

### **📁 Data Protection**
- **Conversation Privacy**: All chat history stored locally and excluded from version control
- **Document Security**: Vector database and embeddings kept local by default
- **Comprehensive .gitignore**: Extensive exclusions prevent accidental commit of sensitive files
- **Multiple Knowledge Bases**: Support for isolated document directories

### **🔍 Automated Security Scanning**

#### **GitLeaks Integration**
The repository includes automated GitLeaks scanning that:
- 🕵️ **Scans every PR and push** for potential secrets and credentials
- 🎯 **Custom rules** for OpenAI, Anthropic, Google, AWS, and GitHub tokens
- 📋 **Smart allowlisting** to prevent false positives on template files
- 🚨 **Automatic alerts** on pull requests if secrets are detected
- 📊 **Detailed reports** uploaded as artifacts for investigation

#### **Scanning Coverage**
- OpenAI API keys (`sk-...`)
- Anthropic API keys (`sk-ant-...`)
- Google API keys (`AIza...`)
- AWS access keys (`AKIA...`)
- GitHub tokens (`ghp_...`)
- Private keys (PEM format)
- Database connection strings
- SSH keys and certificates

### **🏗️ Architecture Security**
- **Local-First Design**: Processes documents locally when possible
- **Secure HTTP**: Web interface binds to localhost by default
- **Input Validation**: Proper sanitization of user inputs
- **Error Handling**: Sensitive information not exposed in error messages

## 🚨 Reporting Security Issues

If you discover a security vulnerability, please follow responsible disclosure:

1. **Do NOT open a public issue**
2. **Email security concerns** to: [your-security-email@domain.com]
3. **Include detailed information**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested mitigation (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 5 business days.

## ✅ Security Checklist for Contributors

Before contributing, ensure:

- [ ] **No hardcoded credentials** in your code
- [ ] **Environment variables** used for all sensitive data
- [ ] **Test credentials** are clearly marked as placeholders
- [ ] **Personal data** not included in commits
- [ ] **Security scan passes** (GitLeaks will run automatically)

## 🔧 Security Configuration

### **Required Environment Variables**
```bash
# External API Configuration (Optional)
EXTERNAL_API_KEY=your_actual_api_key_here
EXTERNAL_API_PROVIDER=openai  # or anthropic, google

# Corporate Portal (Optional)
CORPORATE_PORTAL_API_KEY=your_portal_key_here
CORPORATE_PORTAL_USERNAME=your_username
CORPORATE_PORTAL_PASSWORD=your_password
```

### **Secure Setup Process**
1. **Clone repository** (no secrets included)
2. **Run setup script**: `./install.sh` creates `.env` template
3. **Add your credentials** to `.env` file
4. **Verify security**: Run `git status` to ensure `.env` is ignored

### **Production Deployment**
- Use **secrets management systems** (AWS Secrets Manager, Azure Key Vault, etc.)
- **Rotate credentials** regularly
- **Monitor access logs** for unusual activity
- **Use HTTPS** for external communications
- **Regular security updates** for dependencies

## 📊 Security Monitoring

### **Automated Checks**
- ✅ **GitLeaks scanning** on every commit
- ✅ **Dependency vulnerability scanning** (planned)
- ✅ **Code quality checks** (planned)
- ✅ **License compliance** (planned)

### **Manual Reviews**
- 🔍 **Security code reviews** for sensitive changes
- 📋 **Periodic security audits** of dependencies
- 🎯 **Penetration testing** for production deployments
- 📊 **Access log analysis** for unusual patterns

## 🛠️ Security Tools Used

- **[GitLeaks](https://github.com/gitleaks/gitleaks)**: Secret scanning and detection
- **[GitHub Security Advisories](https://github.com/advisories)**: Dependency vulnerability monitoring
- **[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)**: Secure configuration management
- **[Python Cryptography](https://cryptography.io/)**: When encryption is needed

## 📚 Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Python Security Best Practices](https://python.org/dev/security/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

## 🔄 Security Updates

This security policy is reviewed and updated regularly. Last updated: **[Current Date]**

For security updates and announcements, watch this repository or check the [Releases](../../releases) page.

---

**Remember**: Security is everyone's responsibility. When in doubt, ask questions and err on the side of caution. 🛡️ 