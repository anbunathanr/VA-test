# test_executor_s3.py
import asyncio
import json
import logging
import uuid
import gzip
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_id: str
    test_name: str
    test_suite: str
    status: str  # PASSED, FAILED, ERROR, SKIPPED
    start_time: datetime
    end_time: datetime
    duration: float
    environment: str
    browser: str
    platform: str
    error_message: Optional[str] = None
    screenshot_url: Optional[str] = None
    video_url: Optional[str] = None
    steps_executed: int = 0
    steps_failed: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TestSuiteResult:
    """Test suite result data structure"""
    suite_id: str
    suite_name: str
    project_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    pass_rate: float
    environment: str
    test_results: List[TestResult]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class S3TestStorage:
    """Handles storing test results and artifacts in AWS S3"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        
        try:
            self.s3_client = boto3.client('s3', region_name=region)
            self.s3_resource = boto3.resource('s3', region_name=region)
            
            # Verify bucket exists
            self._ensure_bucket_exists()
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' verified")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' does not exist")
                raise
            else:
                logger.error(f"Error accessing S3 bucket: {str(e)}")
                raise

    def _generate_s3_key(self, project_id: str, suite_id: str, file_type: str, file_name: str) -> str:
        """Generate S3 key for organized storage"""
        date_path = datetime.now(timezone.utc).strftime('%Y/%m/%d')
        return f"test-results/{project_id}/{date_path}/{suite_id}/{file_type}/{file_name}"

    async def store_test_suite_result(self, suite_result: TestSuiteResult) -> str:
        """Store complete test suite result in S3"""
        try:
            # Convert to JSON
            suite_data = {
                'suite_info': asdict(suite_result),
                'test_results': [asdict(test) for test in suite_result.test_results],
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'storage_version': '1.0'
            }
            
            # Convert datetime objects to ISO strings
            suite_data = self._serialize_datetime_objects(suite_data)
            
            # Compress the data
            json_data = json.dumps(suite_data, indent=2)
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            # Generate S3 key
            file_name = f"{suite_result.suite_id}_results.json.gz"
            s3_key = self._generate_s3_key(
                suite_result.project_id,
                suite_result.suite_id,
                'results',
                file_name
            )
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=compressed_data,
                ContentType='application/json',
                ContentEncoding='gzip',
                Metadata={
                    'project-id': suite_result.project_id,
                    'suite-id': suite_result.suite_id,
                    'user-id': suite_result.user_id,
                    'environment': suite_result.environment,
                    'test-count': str(suite_result.total_tests),
                    'pass-rate': str(suite_result.pass_rate)
                }
            )
            
            logger.info(f"Test suite result stored at s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to store test suite result: {str(e)}")
            raise

    async def store_test_artifact(self, project_id: str, suite_id: str, 
                                artifact_type: str, file_name: str, 
                                file_data: bytes, content_type: str = None) -> str:
        """Store test artifacts (screenshots, videos, logs) in S3"""
        try:
            s3_key = self._generate_s3_key(project_id, suite_id, artifact_type, file_name)
            
            put_args = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_data,
                'Metadata': {
                    'project-id': project_id,
                    'suite-id': suite_id,
                    'artifact-type': artifact_type
                }
            }
            
            if content_type:
                put_args['ContentType'] = content_type
            
            self.s3_client.put_object(**put_args)
            
            # Generate public URL (if bucket allows public read)
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Artifact stored at s3://{self.bucket_name}/{s3_key}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to store artifact: {str(e)}")
            raise

    async def store_logs(self, project_id: str, suite_id: str, logs: str) -> str:
        """Store test execution logs in S3"""
        try:
            file_name = f"{suite_id}_logs.txt"
            log_data = logs.encode('utf-8')
            
            return await self.store_test_artifact(
                project_id, suite_id, 'logs', file_name, log_data, 'text/plain'
            )
            
        except Exception as e:
            logger.error(f"Failed to store logs: {str(e)}")
            raise

    def _serialize_datetime_objects(self, obj):
        """Recursively convert datetime objects to ISO strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._serialize_datetime_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        else:
            return obj

    async def get_test_results(self, project_id: str, days: int = 30) -> List[Dict]:
        """Retrieve test results for a project from the last N days"""
        try:
            prefix = f"test-results/{project_id}/"
            
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            results = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('_results.json.gz'):
                        # Check if within date range
                        if (datetime.now(timezone.utc) - obj['LastModified']).days <= days:
                            # Download and parse
                            try:
                                obj_response = self.s3_client.get_object(
                                    Bucket=self.bucket_name,
                                    Key=obj['Key']
                                )
                                
                                # Decompress and parse
                                compressed_data = obj_response['Body'].read()
                                json_data = gzip.decompress(compressed_data).decode('utf-8')
                                result_data = json.loads(json_data)
                                
                                results.append(result_data)
                                
                            except Exception as e:
                                logger.warning(f"Failed to parse result file {obj['Key']}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve test results: {str(e)}")
            raise

class TestReportGenerator:
    """Generate various reports from test results"""
    
    def __init__(self, s3_storage: S3TestStorage):
        self.s3_storage = s3_storage

    async def generate_dashboard_data(self, project_id: str, days: int = 30) -> Dict:
        """Generate dashboard data for SaaS application"""
        try:
            # Get test results
            results = await self.s3_storage.get_test_results(project_id, days)
            
            if not results:
                return self._empty_dashboard_data()
            
            # Process results
            dashboard_data = {
                'summary': self._calculate_summary_metrics(results),
                'trends': self._calculate_trend_data(results),
                'test_distribution': self._calculate_test_distribution(results),
                'environment_breakdown': self._calculate_environment_breakdown(results),
                'recent_failures': self._get_recent_failures(results),
                'performance_metrics': self._calculate_performance_metrics(results),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {str(e)}")
            raise

    def _empty_dashboard_data(self) -> Dict:
        """Return empty dashboard data structure"""
        return {
            'summary': {
                'total_test_runs': 0,
                'total_tests': 0,
                'overall_pass_rate': 0,
                'avg_execution_time': 0,
                'active_environments': 0
            },
            'trends': [],
            'test_distribution': {},
            'environment_breakdown': {},
            'recent_failures': [],
            'performance_metrics': {},
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    def _calculate_summary_metrics(self, results: List[Dict]) -> Dict:
        """Calculate summary metrics"""
        total_runs = len(results)
        total_tests = sum(r['suite_info']['total_tests'] for r in results)
        total_passed = sum(r['suite_info']['passed_tests'] for r in results)
        
        avg_pass_rate = sum(r['suite_info']['pass_rate'] for r in results) / total_runs if total_runs > 0 else 0
        
        # Calculate average execution time
        total_duration = 0
        for result in results:
            start = datetime.fromisoformat(result['suite_info']['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(result['suite_info']['end_time'].replace('Z', '+00:00'))
            total_duration += (end - start).total_seconds()
        
        avg_execution_time = total_duration / total_runs if total_runs > 0 else 0
        
        # Count unique environments
        environments = set(r['suite_info']['environment'] for r in results)
        
        return {
            'total_test_runs': total_runs,
            'total_tests': total_tests,
            'overall_pass_rate': round(avg_pass_rate, 2),
            'avg_execution_time': round(avg_execution_time, 2),
            'active_environments': len(environments)
        }

    def _calculate_trend_data(self, results: List[Dict]) -> List[Dict]:
        """Calculate trend data for charts"""
        # Group by date
        daily_data = {}
        
        for result in results:
            date = result['suite_info']['start_time'][:10]  # Extract date part
            
            if date not in daily_data:
                daily_data[date] = {
                    'date': date,
                    'test_runs': 0,
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'pass_rate': 0
                }
            
            daily_data[date]['test_runs'] += 1
            daily_data[date]['total_tests'] += result['suite_info']['total_tests']
            daily_data[date]['passed_tests'] += result['suite_info']['passed_tests']
            daily_data[date]['failed_tests'] += result['suite_info']['failed_tests']
        
        # Calculate pass rates
        for data in daily_data.values():
            if data['total_tests'] > 0:
                data['pass_rate'] = round((data['passed_tests'] / data['total_tests']) * 100, 2)
        
        return sorted(daily_data.values(), key=lambda x: x['date'])

    def _calculate_test_distribution(self, results: List[Dict]) -> Dict:
        """Calculate test distribution by status"""
        distribution = {
            'passed': 0,
            'failed': 0,
            'error': 0,
            'skipped': 0
        }
        
        for result in results:
            distribution['passed'] += result['suite_info']['passed_tests']
            distribution['failed'] += result['suite_info']['failed_tests']
            distribution['error'] += result['suite_info']['error_tests']
            distribution['skipped'] += result['suite_info']['skipped_tests']
        
        return distribution

    def _calculate_environment_breakdown(self, results: List[Dict]) -> Dict:
        """Calculate breakdown by environment"""
        env_data = {}
        
        for result in results:
            env = result['suite_info']['environment']
            if env not in env_data:
                env_data[env] = {
                    'test_runs': 0,
                    'total_tests': 0,
                    'passed_tests': 0,
                    'pass_rate': 0
                }
            
            env_data[env]['test_runs'] += 1
            env_data[env]['total_tests'] += result['suite_info']['total_tests']
            env_data[env]['passed_tests'] += result['suite_info']['passed_tests']
        
        # Calculate pass rates
        for data in env_data.values():
            if data['total_tests'] > 0:
                data['pass_rate'] = round((data['passed_tests'] / data['total_tests']) * 100, 2)
        
        return env_data

    def _get_recent_failures(self, results: List[Dict], limit: int = 10) -> List[Dict]:
        """Get recent test failures"""
        failures = []
        
        for result in results:
            for test in result['test_results']:
                if test['status'] in ['FAILED', 'ERROR']:
                    failures.append({
                        'test_name': test['test_name'],
                        'suite_name': result['suite_info']['suite_name'],
                        'status': test['status'],
                        'error_message': test.get('error_message', ''),
                        'timestamp': test['start_time'],
                        'environment': result['suite_info']['environment']
                    })
        
        # Sort by timestamp and limit
        failures.sort(key=lambda x: x['timestamp'], reverse=True)
        return failures[:limit]

    def _calculate_performance_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        durations = []
        
        for result in results:
            start = datetime.fromisoformat(result['suite_info']['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(result['suite_info']['end_time'].replace('Z', '+00:00'))
            duration = (end - start).total_seconds()
            durations.append(duration)
        
        if not durations:
            return {}
        
        return {
            'min_duration': round(min(durations), 2),
            'max_duration': round(max(durations), 2),
            'avg_duration': round(sum(durations) / len(durations), 2),
            'median_duration': round(sorted(durations)[len(durations) // 2], 2)
        }

    async def generate_html_report(self, project_id: str, days: int = 30) -> str:
        """Generate HTML report"""
        dashboard_data = await self.generate_dashboard_data(project_id, days)
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics { display: flex; gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; flex: 1; }
                .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
                .metric-label { color: #666; margin-top: 5px; }
                .chart-container { background: white; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .failures-table { width: 100%; border-collapse: collapse; }
                .failures-table th, .failures-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .failures-table th { background-color: #f2f2f2; }
                .status-failed { color: #dc3545; font-weight: bold; }
                .status-error { color: #fd7e14; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Execution Report</h1>
                <p>Project ID: {project_id}</p>
                <p>Generated: {generated_at}</p>
                <p>Period: Last {days} days</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{total_test_runs}</div>
                    <div class="metric-label">Test Runs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_tests}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{overall_pass_rate}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_execution_time}s</div>
                    <div class="metric-label">Avg Duration</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Test Results Distribution</h3>
                <canvas id="distributionChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Recent Failures</h3>
                <table class="failures-table">
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Suite</th>
                            <th>Status</th>
                            <th>Environment</th>
                            <th>Error</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
                        {failures_rows}
                    </tbody>
                </table>
            </div>
            
            <script>
                // Distribution Chart
                const ctx = document.getElementById('distributionChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Passed', 'Failed', 'Error', 'Skipped'],
                        datasets: [{{
                            data: [{passed}, {failed}, {error}, {skipped}],
                            backgroundColor: ['#28a745', '#dc3545', '#fd7e14', '#6c757d']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Generate failures table rows
        failures_rows = ""
        for failure in dashboard_data['recent_failures']:
            status_class = f"status-{failure['status'].lower()}"
            failures_rows += f"""
                <tr>
                    <td>{failure['test_name']}</td>
                    <td>{failure['suite_name']}</td>
                    <td class="{status_class}">{failure['status']}</td>
                    <td>{failure['environment']}</td>
                    <td>{failure['error_message'][:100]}...</td>
                    <td>{failure['timestamp']}</td>
                </tr>
            """
        
        # Format the HTML
        html_report = html_template.format(
            project_id=project_id,
            generated_at=dashboard_data['generated_at'],
            days=days,
            total_test_runs=dashboard_data['summary']['total_test_runs'],
            total_tests=dashboard_data['summary']['total_tests'],
            overall_pass_rate=dashboard_data['summary']['overall_pass_rate'],
            avg_execution_time=dashboard_data['summary']['avg_execution_time'],
            passed=dashboard_data['test_distribution']['passed'],
            failed=dashboard_data['test_distribution']['failed'],
            error=dashboard_data['test_distribution']['error'],
            skipped=dashboard_data['test_distribution']['skipped'],
            failures_rows=failures_rows
        )
        
        return html_report

class SaaSTestExecutor:
    """Main test executor for SaaS application"""
    
    def __init__(self, s3_bucket: str, aws_region: str = 'us-east-1'):
        self.s3_storage = S3TestStorage(s3_bucket, aws_region)
        self.report_generator = TestReportGenerator(self.s3_storage)
        self.current_logs = StringIO()
        
        # Setup logging to capture logs
        self.log_handler = logging.StreamHandler(self.current_logs)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.log_handler)

    async def execute_test_suite(self, project_id: str, user_id: str, 
                               suite_name: str, test_functions: List[callable],
                               environment: str = 'production') -> TestSuiteResult:
        """Execute a test suite and store results in S3"""
        
        suite_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting test suite execution: {suite_name} (ID: {suite_id})")
        
        test_results = []
        passed_count = 0
        failed_count = 0
        error_count = 0
        skipped_count = 0
        
        # Execute each test
        for i, test_func in enumerate(test_functions):
            test_id = str(uuid.uuid4())
            test_start = datetime.now(timezone.utc)
            
            try:
                logger.info(f"Executing test {i+1}/{len(test_functions)}: {test_func.__name__}")
                
                # Execute the test function
                result = await self._execute_single_test(test_func, test_id, environment)
                
                if result.status == 'PASSED':
                    passed_count += 1
                elif result.status == 'FAILED':
                    failed_count += 1
                elif result.status == 'ERROR':
                    error_count += 1
                else:
                    skipped_count += 1
                
                test_results.append(result)
                
            except Exception as e:
                logger.error(f"Test execution failed: {str(e)}")
                
                # Create error result
                error_result = TestResult(
                    test_id=test_id,
                    test_name=test_func.__name__,
                    test_suite=suite_name,
                    status='ERROR',
                    start_time=test_start,
                    end_time=datetime.now(timezone.utc),
                    duration=(datetime.now(timezone.utc) - test_start).total_seconds(),
                    environment=environment,
                    browser='unknown',
                    platform='unknown',
                    error_message=str(e)
                )
                
                test_results.append(error_result)
                error_count += 1
        
        end_time = datetime.now(timezone.utc)
        total_tests = len(test_functions)
        pass_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        # Create suite result
        suite_result = TestSuiteResult(
            suite_id=suite_id,
            suite_name=suite_name,
            project_id=project_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_count,
            failed_tests=failed_count,
            error_tests=error_count,
            skipped_tests=skipped_count,
            pass_rate=round(pass_rate, 2),
            environment=environment,
            test_results=test_results
        )
        
        # Store results in S3
        try:
            await self.s3_storage.store_test_suite_result(suite_result)
            
            # Store logs
            logs_content = self.current_logs.getvalue()
            if logs_content:
                await self.s3_storage.store_logs(project_id, suite_id, logs_content)
            
            logger.info(f"Test suite execution completed: {suite_name}")
            logger.info(f"Results: {passed_count} passed, {failed_count} failed, {error_count} errors")
            
        except Exception as e:
            logger.error(f"Failed to store test results: {str(e)}")
            raise
        
        return suite_result

    async def _execute_single_test(self, test_func: callable, test_id: str, 
                                 environment: str) -> TestResult:
        """Execute a single test function"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Execute test function
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            
            # Test passed
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name=test_func.__name__,
                test_suite='',  # Will be set by caller
                status='PASSED',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                environment=environment,
                browser='chrome',  # Default
                platform='linux',  # Default
                steps_executed=1,
                assertions_passed=1
            )
            
        except AssertionError as e:
            # Test failed
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name=test_func.__name__,
                test_suite='',
                status='FAILED',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                environment=environment,
                browser='chrome',
                platform='linux',
                error_message=str(e),
                steps_executed=1,
                assertions_failed=1
            )
            
        except Exception as e:
            # Test error
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_id,
                test_name=test_func.__name__,
                test_suite='',
                status='ERROR',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                environment=environment,
                browser='chrome',
                platform='linux',
                error_message=str(e)
            )

    async def get_dashboard_data(self, project_id: str, days: int = 30) -> Dict:
        """Get dashboard data for SaaS application"""
        return await self.report_generator.generate_dashboard_data(project_id, days)

    async def generate_report(self, project_id: str, days: int = 30, 
                            format: str = 'html') -> str:
        """Generate test report"""
        if format.lower() == 'html':
            return await self.report_generator.generate_html_report(project_id, days)
        else:
            # Return JSON data
            dashboard_data = await self.report_generator.generate_dashboard_data(project_id, days)
            return json.dumps(dashboard_data, indent=2)

# Example test functions
async def test_login_functionality():
    """Example test function"""
    # Simulate test logic
    await asyncio.sleep(1)  # Simulate test execution time
    
    # Simulate assertion
    assert True, "Login should work correctly"

async def test_user_registration():
    """Example test function"""
    await asyncio.sleep(2)
    assert True, "User registration should work"

def test_api_endpoint():
    """Example synchronous test function"""
    # Simulate API test
    import time
    time.sleep(1)
    
    # This test will fail for demonstration
    assert False, "API endpoint returned unexpected response"

async def test_database_connection():
    """Example test function"""
    await asyncio.sleep(0.5)
    assert True, "Database connection should be successful"

# Example usage
async def main():
    """Example usage of the SaaS test executor"""
    
    # Initialize executor
    executor = SaaSTestExecutor(
        s3_bucket='your-test-results-bucket',
        aws_region='us-east-1'
    )
    
    # Define test suite
    test_functions = [
        test_login_functionality,
        test_user_registration,
        test_api_endpoint,  # This will fail
        test_database_connection
    ]
    
    try:
        # Execute test suite
        suite_result = await executor.execute_test_suite(
            project_id='project-123',
            user_id='user-456',
            suite_name='Core Functionality Tests',
            test_functions=test_functions,
            environment='staging'
        )
        
        print(f"Test suite completed: {suite_result.suite_name}")
        print(f"Results: {suite_result.passed_tests}/{suite_result.total_tests} passed")
        print(f"Pass rate: {suite_result.pass_rate}%")
        
        # Generate dashboard data
        dashboard_data = await executor.get_dashboard_data('project-123')
        print(f"\nDashboard Summary:")
        print(f"Total test runs: {dashboard_data['summary']['total_test_runs']}")
        print(f"Overall pass rate: {dashboard_data['summary']['overall_pass_rate']}%")
        
        # Generate HTML report
        html_report = await executor.generate_report('project-123', format='html')
        
        # Save report locally for viewing
        with open('test_report.html', 'w') as f:
            f.write(html_report)
        
        print("\nHTML report generated: test_report.html")
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
