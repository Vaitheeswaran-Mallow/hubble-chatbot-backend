import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re


class EndorsementDataProcessor:
    """Processes the endorsement process Excel file for RAG system"""
    
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.df = None
        self.processed_data = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the Excel file and return cleaned DataFrame"""
        try:
            self.df = pd.read_excel(self.excel_file_path)
            return self.df
        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")
    
    def clean_data(self) -> List[Dict[str, Any]]:
        """Clean and structure the data for RAG processing"""
        if self.df is None:
            self.load_data()
        
        processed_records = []
        
        for index, row in self.df.iterrows():
            # Skip empty rows and header rows
            if pd.isna(row['S. No']) or row['S. No'] == 'S. No':
                continue
                
            # Skip rows with NaN in Management of column (these are continuation rows)
            if pd.isna(row['Management of']):
                continue
            
            # Create a comprehensive text representation of each process
            process_text = self._create_process_text(row)
            
            if process_text.strip():  # Only add non-empty processes
                processed_records.append({
                    'id': int(row['S. No']) if not pd.isna(row['S. No']) else index,
                    'management_of': str(row['Management of']).strip(),
                    'actions': str(row['Actions']).strip() if not pd.isna(row['Actions']) else '',
                    'initiator': str(row['Initiator']).strip() if not pd.isna(row['Initiator']) else '',
                    'approver': str(row['Approver']).strip() if not pd.isna(row['Approver']) else '',
                    'executer': str(row['Executer']).strip() if not pd.isna(row['Executer']) else '',
                    'post_execution_confirmation': str(row['Post Execution Confirmation']).strip() if not pd.isna(row['Post Execution Confirmation']) else '',
                    'email_details': str(row['Email Details']).strip() if not pd.isna(row['Email Details']) else '',
                    'full_text': process_text,
                    'metadata': {
                        'row_index': index,
                        'has_email_template': bool(row['Email Details']) and not pd.isna(row['Email Details']),
                        'process_type': self._categorize_process(row['Management of'])
                    }
                })
        
        self.processed_data = processed_records
        return processed_records
    
    def _create_process_text(self, row: pd.Series) -> str:
        """Create a comprehensive text representation of a process"""
        text_parts = []
        
        # Add management area
        if not pd.isna(row['Management of']):
            text_parts.append(f"Process: {row['Management of']}")
        
        # Add actions
        if not pd.isna(row['Actions']):
            text_parts.append(f"Actions: {row['Actions']}")
        
        # Add initiator
        if not pd.isna(row['Initiator']):
            text_parts.append(f"Who initiates: {row['Initiator']}")
        
        # Add approver
        if not pd.isna(row['Approver']):
            text_parts.append(f"Who approves: {row['Approver']}")
        
        # Add executer
        if not pd.isna(row['Executer']):
            text_parts.append(f"Who executes: {row['Executer']}")
        
        # Add post execution confirmation
        if not pd.isna(row['Post Execution Confirmation']):
            text_parts.append(f"Post execution confirmation: {row['Post Execution Confirmation']}")
        
        # Add email details
        if not pd.isna(row['Email Details']):
            text_parts.append(f"Email template: {row['Email Details']}")
        
        return " | ".join(text_parts)
    
    def _categorize_process(self, management_of: str) -> str:
        """Categorize the process type based on management area"""
        if pd.isna(management_of):
            return "unknown"
        
        management_lower = str(management_of).lower()
        
        if any(keyword in management_lower for keyword in ['application', 'software', 'install', 'upgrade']):
            return "software_management"
        elif any(keyword in management_lower for keyword in ['network', 'restriction', 'access']):
            return "network_access"
        elif any(keyword in management_lower for keyword in ['email', 'mailbox', 'forwarder']):
            return "email_management"
        elif any(keyword in management_lower for keyword in ['bitbucket', 'repository', 'aws', 'firebase']):
            return "development_tools"
        elif any(keyword in management_lower for keyword in ['certification', 'purchase', 'finance']):
            return "finance_certification"
        elif any(keyword in management_lower for keyword in ['team', 'work', 'schedule', 'location']):
            return "hr_management"
        else:
            return "general"
    
    def get_processes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all processes of a specific category"""
        return [process for process in self.processed_data if process['metadata']['process_type'] == category]
    
    def search_processes(self, query: str) -> List[Dict[str, Any]]:
        """Simple text search in processes"""
        query_lower = query.lower()
        matching_processes = []
        
        for process in self.processed_data:
            # Search in full text
            if query_lower in process['full_text'].lower():
                matching_processes.append(process)
            # Search in specific fields
            elif (query_lower in process['management_of'].lower() or 
                  query_lower in process['actions'].lower() or
                  query_lower in process['initiator'].lower() or
                  query_lower in process['approver'].lower()):
                matching_processes.append(process)
        
        return matching_processes
    
    def get_all_categories(self) -> List[str]:
        """Get all available process categories"""
        categories = set()
        for process in self.processed_data:
            categories.add(process['metadata']['process_type'])
        return list(categories)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the processed data"""
        if not self.processed_data:
            return {}
        
        categories = {}
        for process in self.processed_data:
            category = process['metadata']['process_type']
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_processes': len(self.processed_data),
            'categories': categories,
            'processes_with_email_templates': sum(1 for p in self.processed_data if p['metadata']['has_email_template']),
            'file_path': self.excel_file_path
        }


if __name__ == "__main__":
    # Test the data processor
    processor = EndorsementDataProcessor('/Users/yuvaraj/Downloads/Mallow Hackathon 2025/Endorsement Process Excel.xlsx')
    processor.load_data()
    processed_data = processor.clean_data()
    
    print(f"Processed {len(processed_data)} processes")
    print("\nSummary:")
    stats = processor.get_summary_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nSample process:")
    if processed_data:
        sample = processed_data[0]
        print(f"ID: {sample['id']}")
        print(f"Management: {sample['management_of']}")
        print(f"Category: {sample['metadata']['process_type']}")
        print(f"Text: {sample['full_text'][:200]}...")
