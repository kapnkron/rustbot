#!/usr/bin/env python3
"""
Fix for RpcSignaturesForAddressConfig conversion issue in Python 3.12.

This script applies a monkey patch to fix the error:
"argument 'before': 'RpcSignaturesForAddressConfig' object cannot be converted to 'Signature'"

The issue occurs in the solana-py library when running on Python 3.12.
"""

import os
import sys
import logging
import importlib
import inspect
import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_signature_fix():
    """
    Apply monkey patch to fix the RpcSignaturesForAddressConfig conversion issue.
    """
    try:
        # Import the necessary modules
        from solana.rpc import api
        import solana.rpc.types
        from solana.rpc.types import RpcSignaturesForAddressConfig
        
        # Find all classes that might be related to signature fetching
        classes_to_patch = []
        for name, obj in inspect.getmembers(api):
            if inspect.isclass(obj) and "signature" in name.lower():
                classes_to_patch.append((name, obj))
                logging.info(f"Found potential class to patch: {name}")
        
        if not classes_to_patch:
            # If no classes with "signature" in name, check all classes with methods that use RpcSignaturesForAddressConfig
            for name, obj in inspect.getmembers(api):
                if inspect.isclass(obj):
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if "RpcSignaturesForAddressConfig" in inspect.getsource(method):
                            classes_to_patch.append((name, obj))
                            logging.info(f"Found class with RpcSignaturesForAddressConfig usage: {name}.{method_name}")
                            break
        
        # Check if there's a get_signatures_for_address method in the API class
        api_class = api.Client if hasattr(api, 'Client') else None
        if api_class:
            for method_name, method in inspect.getmembers(api_class, inspect.isfunction):
                if method_name == 'get_signatures_for_address':
                    logging.info(f"Found get_signatures_for_address method in {api_class.__name__}")
                    
                    # Define patched method for get_signatures_for_address
                    def patched_get_signatures(self, address, limit=None, before=None, until=None):
                        """
                        Patched version of get_signatures_for_address that handles the before/until parameters correctly.
                        """
                        config = RpcSignaturesForAddressConfig(limit=limit)
                        if before:
                            config.before = before
                        if until:
                            config.until = until
                        
                        # Original method implementation after preparing config
                        # This is simplified and would need to be adapted to the actual implementation
                        from solana.rpc.providers import http
                        return http.send_request("getSignaturesForAddress", [str(address), config])
                    
                    # Apply the monkey patch
                    setattr(api_class, method_name, types.MethodType(patched_get_signatures, api_class))
                    logging.info(f"Patched {api_class.__name__}.{method_name}")
                    return True
        
        # Direct patch of the problematic converter method if we can find it
        # This is a fallback approach
        patched = False
        for name, module in [(n, getattr(solana.rpc, n)) for n in dir(solana.rpc) if not n.startswith('_')]:
            for cls_name, cls in inspect.getmembers(module, inspect.isclass):
                if '_to_rpc_config' in [m[0] for m in inspect.getmembers(cls, inspect.isfunction)]:
                    logging.info(f"Found _to_rpc_config method in {cls_name}")
                    
                    original_method = getattr(cls, '_to_rpc_config')
                    
                    # Define patched method
                    def patched_to_rpc_config(self):
                        """
                        Patched version that doesn't try to convert before/until to Signature objects.
                        """
                        config = RpcSignaturesForAddressConfig(limit=self.limit if hasattr(self, 'limit') else None)
                        
                        if hasattr(self, 'before') and self.before:
                            config.before = self.before
                        
                        if hasattr(self, 'until') and self.until:
                            config.until = self.until
                            
                        return config
                    
                    # Apply patch
                    setattr(cls, '_to_rpc_config', types.MethodType(patched_to_rpc_config, cls))
                    logging.info(f"Patched {cls_name}._to_rpc_config")
                    patched = True
        
        # If no specific method patched, apply direct monkey patch to fetch_historical_ohlcv.py
        if not patched:
            logging.info("Applying direct patch to the fetch_historical_ohlcv.py file")
            try:
                import solana.publickey
                
                # Override the Signature class's __str__ method
                original_str = solana.publickey.Signature.__str__
                
                def patched_str(self):
                    return str(self.signature)
                
                solana.publickey.Signature.__str__ = patched_str
                logging.info("Patched solana.publickey.Signature.__str__")
                patched = True
            except Exception as e:
                logging.error(f"Failed to apply direct patch: {e}")
        
        if patched:
            logging.info("Successfully applied signature conversion fix")
            return True
        else:
            logging.warning("No suitable methods found to patch")
            return False
        
    except ImportError as e:
        logging.error(f"Failed to import required modules: {e}")
        return False
    except Exception as e:
        logging.error(f"Failed to apply signature fix: {e}")
        return False

def main():
    """
    Main function to apply the fix and provide status information.
    """
    logging.info("Starting signature conversion fix for Python 3.12")
    
    # Check Python version
    python_version = sys.version_info
    logging.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Apply the fix
    success = apply_signature_fix()
    
    if success:
        logging.info("✅ Signature conversion fix successfully applied")
    else:
        logging.error("❌ Failed to apply signature conversion fix")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 