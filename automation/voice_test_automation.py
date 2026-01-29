# test_automation_framework.py
import re
import json
import asyncio
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Playwright imports
from playwright.async_api import async_playwright, Page as PlaywrightPage, Browser as PlaywrightBrowser

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrowserType(Enum):
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"

@dataclass
class TestAction:
    action_type: str
    element: Optional[str] = None
    value: Optional[str] = None
    expected_result: Optional[str] = None
    timeout: int = 10000

@dataclass
class TestCase:
    name: str
    description: str
    actions: List[TestAction]
    setup_url: Optional[str] = None

class NLPTestParser:
    """Natural Language Processing for test instructions"""
    
    def __init__(self):
        self.action_patterns = {
            # Navigation patterns
            r'(?:go to|navigate to|visit|open)\s+(.+?)(?:\s|$)': 'navigate',
            r'(?:click|press|tap)\s+(?:on\s+)?(.+?)(?:\s|$)': 'click',
            r'(?:type|enter|input)\s+["\'](.+?)["\'](?:\s+(?:in|into|on)\s+(.+?))?(?:\s|$)': 'type',
            r'(?:fill|populate)\s+(.+?)\s+(?:with|using)\s+["\'](.+?)["\']': 'fill',
            r'(?:select|choose)\s+["\'](.+?)["\'](?:\s+(?:from|in)\s+(.+?))?': 'select',
            r'(?:wait for|expect)\s+(.+?)(?:\s+to\s+(?:appear|be visible|exist))?': 'wait_for',
            r'(?:verify|check|assert)\s+(?:that\s+)?(.+?)(?:\s+(?:is|contains|equals)\s+["\'](.+?)["\'])?': 'verify',
            r'(?:login|log in|sign in)\s+(?:with|using)\s+(.+?)': 'login',
            r'(?:logout|log out|sign out)': 'logout',
            r'(?:scroll\s+)?(?:down|up)(?:\s+to\s+(.+?))?': 'scroll',
            r'(?:take|capture)\s+(?:a\s+)?screenshot': 'screenshot',
            r'(?:refresh|reload)\s+(?:the\s+)?page': 'refresh'
        }
        
        self.element_selectors = {
            'login button': 'button[type="submit"], input[type="submit"], .login-btn, #login, [data-testid*="login"]',
            'username': 'input[name="username"], input[name="email"], input[type="email"], #username, #email',
            'password': 'input[name="password"], input[type="password"], #password',
            'submit button': 'button[type="submit"], input[type="submit"], .submit-btn',
            'search box': 'input[type="search"], input[name="search"], .search-input, #search',
            'menu': '.menu, .nav, nav, .navigation',
            'dropdown': 'select, .dropdown, .select',
            'checkbox': 'input[type="checkbox"]',
            'radio button': 'input[type="radio"]'
        }

    def parse_instruction(self, instruction: str) -> List[TestAction]:
        """Parse natural language instruction into test actions"""
        instruction = instruction.lower().strip()
        actions = []
        
        # Handle compound instructions (split by 'and', 'then', etc.)
        sub_instructions = re.split(r'\s+(?:and|then|after that|next)\s+', instruction)
        
        for sub_instruction in sub_instructions:
            action = self._parse_single_instruction(sub_instruction.strip())
            if action:
                actions.append(action)
        
        return actions

    def _parse_single_instruction(self, instruction: str) -> Optional[TestAction]:
        """Parse a single instruction into a test action"""
        
        for pattern, action_type in self.action_patterns.items():
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                return self._create_action(action_type, match, instruction)
        
        # Default fallback
        logger.warning(f"Could not parse instruction: {instruction}")
        return TestAction(action_type="manual", element=instruction)

    def _create_action(self, action_type: str, match, instruction: str) -> TestAction:
        """Create a TestAction based on the action type and regex match"""
        
        if action_type == 'navigate':
            url = match.group(1).strip()
            return TestAction(action_type='navigate', value=url)
        
        elif action_type == 'click':
            element = match.group(1).strip()
            selector = self._get_selector(element)
            return TestAction(action_type='click', element=selector)
        
        elif action_type == 'type':
            text = match.group(1)
            element = match.group(2) if match.lastindex > 1 else 'input'
            selector = self._get_selector(element) if element else 'input'
            return TestAction(action_type='type', element=selector, value=text)
        
        elif action_type == 'fill':
            element = match.group(1).strip()
            value = match.group(2).strip()
            selector = self._get_selector(element)
            return TestAction(action_type='fill', element=selector, value=value)
        
        elif action_type == 'login':
            credentials = match.group(1).strip()
            return TestAction(action_type='login', value=credentials)
        
        elif action_type == 'verify':
            element = match.group(1).strip()
            expected = match.group(2) if match.lastindex > 1 else None
            selector = self._get_selector(element)
            return TestAction(action_type='verify', element=selector, expected_result=expected)
        
        elif action_type == 'wait_for':
            element = match.group(1).strip()
            selector = self._get_selector(element)
            return TestAction(action_type='wait_for', element=selector)
        
        else:
            return TestAction(action_type=action_type, element=match.group(1) if match.lastindex >= 1 else None)

    def _get_selector(self, element_description: str) -> str:
        """Convert element description to CSS selector"""
        element_description = element_description.lower().strip()
        
        # Check predefined selectors
        for key, selector in self.element_selectors.items():
            if key in element_description:
                return selector
        
        # Generate selector based on common patterns
        if 'button' in element_description:
            text = re.search(r'button\s+["\'](.+?)["\']', element_description)
            if text:
                return f'button:contains("{text.group(1)}")'
            return 'button'
        
        if 'link' in element_description:
            text = re.search(r'link\s+["\'](.+?)["\']', element_description)
            if text:
                return f'a:contains("{text.group(1)}")'
            return 'a'
        
        # Try to extract text content for generic elements
        text_match = re.search(r'["\'](.+?)["\']', element_description)
        if text_match:
            text = text_match.group(1)
            return f'*:contains("{text}")'
        
        # Fallback to element description as selector
        return element_description

class BrowserAutomation(ABC):
    """Abstract base class for browser automation"""
    
    @abstractmethod
    async def setup(self, headless: bool = False):
        pass
    
    @abstractmethod
    async def teardown(self):
        pass
    
    @abstractmethod
    async def execute_action(self, action: TestAction):
        pass

class PlaywrightAutomation(BrowserAutomation):
    """Playwright implementation of browser automation"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.context = None

    async def setup(self, headless: bool = False):
        """Setup Playwright browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
        # Set default timeout
        self.page.set_default_timeout(30000)
        
        logger.info("Playwright browser setup complete")

    async def teardown(self):
        """Cleanup Playwright resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        logger.info("Playwright browser teardown complete")

    async def execute_action(self, action: TestAction):
        """Execute a test action using Playwright"""
        logger.info(f"Executing action: {action.action_type}")
        
        try:
            if action.action_type == 'navigate':
                await self.page.goto(action.value)
                await self.page.wait_for_load_state('networkidle')
            
            elif action.action_type == 'click':
                await self.page.click(action.element, timeout=action.timeout)
            
            elif action.action_type == 'type' or action.action_type == 'fill':
                await self.page.fill(action.element, action.value)
            
            elif action.action_type == 'wait_for':
                await self.page.wait_for_selector(action.element, timeout=action.timeout)
            
            elif action.action_type == 'verify':
                element = await self.page.wait_for_selector(action.element, timeout=action.timeout)
                if action.expected_result:
                    text = await element.text_content()
                    assert action.expected_result.lower() in text.lower(), f"Expected '{action.expected_result}' not found in '{text}'"
                else:
                    assert element is not None, f"Element '{action.element}' not found"
            
            elif action.action_type == 'login':
                await self._handle_login(action.value)
            
            elif action.action_type == 'screenshot':
                await self.page.screenshot(path=f'screenshot_{int(asyncio.get_event_loop().time())}.png')
            
            elif action.action_type == 'scroll':
                await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            
            elif action.action_type == 'refresh':
                await self.page.reload()
            
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
        
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {str(e)}")
            raise

    async def _handle_login(self, credentials: str):
        """Handle login with credentials"""
        # Parse credentials (format: "username:password" or "valid credentials")
        if ':' in credentials:
            username, password = credentials.split(':', 1)
        else:
            # Use default test credentials
            username, password = "testuser@example.com", "testpassword123"
        
        # Fill username
        username_selectors = ['input[name="username"]', 'input[name="email"]', 'input[type="email"]', '#username', '#email']
        for selector in username_selectors:
            try:
                await self.page.fill(selector, username, timeout=5000)
                break
            except:
                continue
        
        # Fill password
        password_selectors = ['input[name="password"]', 'input[type="password"]', '#password']
        for selector in password_selectors:
            try:
                await self.page.fill(selector, password, timeout=5000)
                break
            except:
                continue
        
        # Click login button
        login_selectors = ['button[type="submit"]', 'input[type="submit"]', '.login-btn', '#login', '[data-testid*="login"]']
        for selector in login_selectors:
            try:
                await self.page.click(selector, timeout=5000)
                break
            except:
                continue

class SeleniumAutomation(BrowserAutomation):
    """Selenium implementation of browser automation"""
    
    def __init__(self):
        self.driver = None

    async def setup(self, headless: bool = False):
        """Setup Selenium WebDriver"""
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)
        
        logger.info("Selenium WebDriver setup complete")

    async def teardown(self):
        """Cleanup Selenium resources"""
        if self.driver:
            self.driver.quit()
        
        logger.info("Selenium WebDriver teardown complete")

    async def execute_action(self, action: TestAction):
        """Execute a test action using Selenium"""
        logger.info(f"Executing action: {action.action_type}")
        
        try:
            wait = WebDriverWait(self.driver, action.timeout // 1000)
            
            if action.action_type == 'navigate':
                self.driver.get(action.value)
            
            elif action.action_type == 'click':
                element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, action.element)))
                element.click()
            
            elif action.action_type == 'type' or action.action_type == 'fill':
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, action.element)))
                element.clear()
                element.send_keys(action.value)
            
            elif action.action_type == 'wait_for':
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, action.element)))
            
            elif action.action_type == 'verify':
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, action.element)))
                if action.expected_result:
                    text = element.text
                    assert action.expected_result.lower() in text.lower(), f"Expected '{action.expected_result}' not found in '{text}'"
                else:
                    assert element is not None, f"Element '{action.element}' not found"
            
            elif action.action_type == 'login':
                await self._handle_login(action.value)
            
            elif action.action_type == 'screenshot':
                self.driver.save_screenshot(f'screenshot_{int(asyncio.get_event_loop().time())}.png')
            
            elif action.action_type == 'scroll':
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            
            elif action.action_type == 'refresh':
                self.driver.refresh()
            
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
        
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {str(e)}")
            raise

    async def _handle_login(self, credentials: str):
        """Handle login with credentials"""
        # Parse credentials
        if ':' in credentials:
            username, password = credentials.split(':', 1)
        else:
            username, password = "testuser@example.com", "testpassword123"
        
        wait = WebDriverWait(self.driver, 10)
        
        # Fill username
        username_selectors = ['input[name="username"]', 'input[name="email"]', 'input[type="email"]', '#username', '#email']
        for selector in username_selectors:
            try:
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                element.clear()
                element.send_keys(username)
                break
            except:
                continue
        
        # Fill password
        password_selectors = ['input[name="password"]', 'input[type="password"]', '#password']
        for selector in password_selectors:
            try:
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                element.clear()
                element.send_keys(password)
                break
            except:
                continue
        
        # Click login button
        login_selectors = ['button[type="submit"]', 'input[type="submit"]', '.login-btn', '#login', '[data-testid*="login"]']
        for selector in login_selectors:
            try:
                element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                element.click()
                break
            except:
                continue

class TestAutomationFramework:
    """Main framework class that orchestrates test execution"""
    
    def __init__(self, browser_type: BrowserType = BrowserType.PLAYWRIGHT):
        self.parser = NLPTestParser()
        self.browser_type = browser_type
        self.automation = None
        
        if browser_type == BrowserType.PLAYWRIGHT:
            self.automation = PlaywrightAutomation()
        else:
            self.automation = SeleniumAutomation()

    async def create_test_from_instruction(self, instruction: str, base_url: str = None) -> TestCase:
        """Create a test case from natural language instruction"""
        actions = self.parser.parse_instruction(instruction)
        
        # Add navigation to base URL if provided and no navigation action exists
        if base_url and not any(action.action_type == 'navigate' for action in actions):
            actions.insert(0, TestAction(action_type='navigate', value=base_url))
        
        test_case = TestCase(
            name=f"Test: {instruction[:50]}...",
            description=instruction,
            actions=actions,
            setup_url=base_url
        )
        
        return test_case

    async def execute_test(self, test_case: TestCase, headless: bool = False) -> Dict:
        """Execute a test case"""
        results = {
            'test_name': test_case.name,
            'status': 'PASSED',
            'actions_executed': 0,
            'actions_failed': 0,
            'errors': [],
            'execution_time': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            await self.automation.setup(headless=headless)
            
            for i, action in enumerate(test_case.actions):
                try:
                    await self.automation.execute_action(action)
                    results['actions_executed'] += 1
                    logger.info(f"Action {i+1}/{len(test_case.actions)} completed successfully")
                    
                    # Add small delay between actions
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    results['actions_failed'] += 1
                    results['errors'].append(f"Action {i+1}: {str(e)}")
                    logger.error(f"Action {i+1} failed: {str(e)}")
                    
                    # Continue with next action or fail the test
                    if action.action_type in ['verify', 'wait_for']:
                        results['status'] = 'FAILED'
                        break
        
        except Exception as e:
            results['status'] = 'ERROR'
            results['errors'].append(f"Setup error: {str(e)}")
            logger.error(f"Test setup failed: {str(e)}")
        
        finally:
            await self.automation.teardown()
            results['execution_time'] = asyncio.get_event_loop().time() - start_time
        
        return results

    async def run_test_from_instruction(self, instruction: str, base_url: str = None, headless: bool = False) -> Dict:
        """Complete workflow: parse instruction and execute test"""
        logger.info(f"Creating test from instruction: {instruction}")
        
        test_case = await self.create_test_from_instruction(instruction, base_url)
        
        logger.info(f"Executing test case with {len(test_case.actions)} actions")
        results = await self.execute_test(test_case, headless=headless)
        
        return results

# Example usage and test runner
async def main():
    """Example usage of the framework"""
    
    # Initialize framework with Playwright (or BrowserType.SELENIUM)
    framework = TestAutomationFramework(BrowserType.PLAYWRIGHT)
    
    # Example test instructions
    test_instructions = [
        "Go to https://example.com and click on the login button",
        "Test login page with valid credentials testuser@example.com:password123",
        "Navigate to https://httpbin.org/forms/post and fill username with 'testuser' and fill password with 'testpass' then click submit",
        "Visit https://www.google.com and type 'playwright automation' in search box and press enter",
        "Go to https://example.com and verify that the page title contains 'Example'"
    ]
    
    # Execute tests
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {instruction}")
        print('='*60)
        
        try:
            results = await framework.run_test_from_instruction(
                instruction=instruction,
                headless=True  # Set to False to see browser actions
            )
            
            print(f"Status: {results['status']}")
            print(f"Actions executed: {results['actions_executed']}")
            print(f"Actions failed: {results['actions_failed']}")
            print(f"Execution time: {results['execution_time']:.2f}s")
            
            if results['errors']:
                print("Errors:")
                for error in results['errors']:
                    print(f"  - {error}")
        
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
