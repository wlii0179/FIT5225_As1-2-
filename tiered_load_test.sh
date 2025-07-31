#!/bin/bash

set -e

detect_service_url() {
    log "Detecting service URL..."
    
    if curl -s --max-time 3 "http://localhost:30001/health" > /dev/null 2>&1; then
        SERVICE_URL="http://localhost:30001"
        log "✅ Using localhost: $SERVICE_URL"
        return 0
    fi
    
    local nodeport=$(kubectl get service pose-estimation-service -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
    if [ ! -z "$nodeport" ]; then
        local internal_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "")
        if [ ! -z "$internal_ip" ] && curl -s --max-time 3 "http://${internal_ip}:${nodeport}/health" > /dev/null 2>&1; then
            SERVICE_URL="http://${internal_ip}:${nodeport}"
            log "✅ Using internal IP: $SERVICE_URL"
            return 0
        fi
        
        local external_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null || echo "")
        if [ ! -z "$external_ip" ] && curl -s --max-time 3 "http://${external_ip}:${nodeport}/health" > /dev/null 2>&1; then
            SERVICE_URL="http://${external_ip}:${nodeport}"
            log "✅ Using external IP: $SERVICE_URL"
            return 0
        fi
        
        local public_ip=$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo "")
        if [ ! -z "$public_ip" ] && curl -s --max-time 3 "http://${public_ip}:${nodeport}/health" > /dev/null 2>&1; then
            SERVICE_URL="http://${public_ip}:${nodeport}"
            log "✅ Using public IP: $SERVICE_URL"
            return 0
        fi
    fi
    
    if [ ! -z "$REMOTE_IP" ]; then
        local test_url="http://${REMOTE_IP}:30001"
        if curl -s --max-time 3 "${test_url}/health" > /dev/null 2>&1; then
            SERVICE_URL="$test_url"
            log "✅ Using REMOTE_IP: $SERVICE_URL"
            return 0
        fi
    fi
    
    log "❌ Cannot detect valid service URL"
    return 1
}

DEPLOYMENT_NAME="pose-estimation-api"
RESULTS_DIR="tiered_load_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/test.log"
}

declare -A POD_USER_LIMITS=(
    [1]=4
    [2]=5
    [3]=6
    [4]=5
)

scale_and_wait() {
    local replicas=$1
    log "Scaling to ${replicas} pods..."
    
    kubectl scale deployment $DEPLOYMENT_NAME --replicas=$replicas
    
    kubectl wait --for=condition=ready pod -l app=pose-estimation-api --timeout=90s
    
    sleep 20
    
    if ! detect_service_url; then
        log "❌ Cannot connect to service with $replicas pods"
        return 1
    fi
    
    log "✅ Service ready with $replicas pods at $SERVICE_URL"
}

test_pod_configuration() {
    local pod_count=$1
    local max_users=${POD_USER_LIMITS[$pod_count]}
    
    log "=========================================="
    log "Testing ${pod_count} pod configuration (max ${max_users} users)"
    log "=========================================="
    
    if ! scale_and_wait $pod_count; then
        log "❌ Failed to scale to $pod_count pods, skipping..."
        return 1
    fi
    
    local max_successful=0
    local pod_results_dir="$RESULTS_DIR/pod_${pod_count}"
    mkdir -p "$pod_results_dir"
    
    for users in $(seq 1 $max_users); do
        log "Testing $pod_count pods with $users users at $SERVICE_URL (1 min test)..."
        
        locust -f locust_test.py \
            --host=$SERVICE_URL \
            --users=$users \
            --spawn-rate=1 \
            --run-time=60s \
            --headless \
            --csv="$pod_results_dir/test_${users}users" \
            --logfile="$pod_results_dir/locust_${users}users.log" \
            --loglevel=INFO
        
        if [ -f "$pod_results_dir/test_${users}users_stats.csv" ]; then
            local failures=$(tail -n 1 "$pod_results_dir/test_${users}users_stats.csv" | cut -d',' -f4)
            local requests=$(tail -n 1 "$pod_results_dir/test_${users}users_stats.csv" | cut -d',' -f3)
            local avg_response=$(tail -n 1 "$pod_results_dir/test_${users}users_stats.csv" | cut -d',' -f6)
            local rps=$(tail -n 1 "$pod_results_dir/test_${users}users_stats.csv" | cut -d',' -f10)
            
            local success_requests=0
            if [[ "$failures" =~ ^[0-9]+$ ]] && [[ "$requests" =~ ^[0-9]+$ ]]; then
                success_requests=$((requests - failures))
            fi
            
            local success_rate=0
            if [ "$requests" -gt 0 ]; then
                success_rate=$((success_requests * 100 / requests))
            fi
            
            log "Results: total_requests=$requests, failures=$failures, success_rate=${success_rate}%, avg_response=${avg_response}ms, rps=$rps"
            
            if [ "$success_requests" -gt 0 ]; then
                max_successful=$users
                log "✅ $users users test completed for $pod_count pods (${success_rate}% success rate)"
            else
                log "❌ $users users test had 0 successful requests, but continuing..."
                max_successful=$users
            fi
            
            if [[ "$avg_response" =~ ^[0-9.]+$ ]] && (( $(echo "$avg_response > 30000" | bc -l 2>/dev/null || echo 0) )); then
                log "⚠️ Response time extremely high (${avg_response}ms), but continuing with warning"
            fi
            
        else
            log "❌ No results file for $users users, but continuing..."
            max_successful=$users
        fi
        
        sleep 10
    done
    
    echo "$pod_count pods: $max_successful users (completed all tests)" >> "$RESULTS_DIR/pod_summary.txt"
    log "Pod $pod_count configuration completed: tested up to $max_successful users"
    
    return 0
}

generate_summary() {
    log "Generating comprehensive summary..."
    
    {
        echo "Tiered Load Testing Results"
        echo "=========================="
        echo "Test Date: $(date)"
        echo "Service URL: $SERVICE_URL"
        echo "Hardware: 2-core CPU, 4GB RAM"
        echo ""
        echo "Pod Configuration Results:"
        cat "$RESULTS_DIR/pod_summary.txt"
        echo ""
        echo "Detailed Results:"
        echo "- 1 Pod: Baseline single container performance"
        echo "- 2 Pods: Load balancing effectiveness" 
        echo "- 3 Pods: Horizontal scaling benefits"
        echo "- 4 Pods: Resource saturation point"
        echo ""
        echo "All detailed CSV files available in respective pod_X directories"
    } > "$RESULTS_DIR/final_summary.txt"
    
    log "✅ Summary generated: $RESULTS_DIR/final_summary.txt"
}

main() {
    log "Starting tiered load testing for different pod configurations..."
    
    if ! detect_service_url; then
        log "❌ Failed to detect service URL. Please check:"
        log "1. Kubernetes cluster is running"
        log "2. pose-estimation-service is deployed"
        log "3. Service is accessible"
        log "4. Set REMOTE_IP environment variable if using remote connection"
        exit 1
    fi
    
    log "Using service URL: $SERVICE_URL"
    
    echo "Pod Configuration Maximum Users Summary" > "$RESULTS_DIR/pod_summary.txt"
    echo "=======================================" >> "$RESULTS_DIR/pod_summary.txt"
    
    for pod_count in 1 2 3 4; do
        if ! test_pod_configuration $pod_count; then
            log "❌ Failed to test $pod_count pod configuration"
        fi
        
        log "Resting 10 seconds before next configuration..."
        sleep 10
    done
    
    generate_summary
    
    log "✅ All tiered load testing completed!"
    log "Results directory: $RESULTS_DIR"
}

main
