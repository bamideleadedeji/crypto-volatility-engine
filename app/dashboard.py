if st.button("🚀 Execute Autonomous Closed-Loop Rebalance", width="stretch"):
        try:
            if not evaluation["rebalance_triggered"]:
                st.info("⏸️ Rebalance skipped: Portfolio drift is within the 3% tolerance threshold. No action required.")
            else:
                st.success("✅ Drift exceeds 3% threshold. Executing Two-Phase Self-Funding Rebalance & Treasury Cash Sweep:")

                position_caps = {"XBTNGN": 0.60, "ETHNGN": 0.40, "SOLNGN": 0.30}
                execution_plan = evaluation["execution_plan"]

                projected_ngn_cash = ngn_balance
                min_order_threshold = 50.0  # Luno Nigeria minimum order size constraint (₦50)

                # --- PHASE 1: DIVESTMENT & CASH REALIZATION (SELLS FIRST) ---
                st.markdown("#### **Phase 1: Divestment & NGN Treasury Sweep (Sells)**")
                phase1_sells = {pair: d for pair, d in execution_plan.items() if d["action"] == "SELL"}

                if not phase1_sells:
                    st.info("No overweight assets to shed in this cycle.")

                for pair, details in phase1_sells.items():
                    target_val = float(details["target_adjustment_val"])
                    current_price = float(prices.get(pair, 0.0))

                    if target_val < min_order_threshold:
                        st.warning(f"⚠️ Skipping SELL for {pair}: Order value ₦{target_val:,.2f} is below Luno minimum exchange threshold.")
                        continue

                    if current_price > 0:
                        if order_executor and not paper_mode:
                            res = safe_execute_market_order(order_executor, pair, "SELL", target_val, current_price)
                            st.write(f"Executed SELL on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash += target_val
                            else:
                                # Graceful handling of Luno balance / dust errors without crashing pipeline
                                err_msg = res.get('error', 'Unknown error')
                                st.warning(f"⚠️ SELL {pair} encountered exchange restriction ({err_msg}). Proceeding with available liquidity sweep.")
                                log_audit(venue="LUNO", asset=pair, price=current_price, amount=target_val, units="FAILED_SELL", status="EXCHANGE_WARNING")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed SELL on {pair} for ₦{target_val:,.2f}. Proceeds swept to NGN wallet.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=target_val, units="SIMULATED", status="PAPER_SELL")
                            projected_ngn_cash += target_val

                # --- PHASE 2: CAPITAL DEPLOYMENT & ACCUMULATION (BUYS SECOND) ---
                st.markdown("#### **Phase 2: Capital Deployment & Accumulation (Buys)**")
                phase2_buys = {pair: d for pair, d in execution_plan.items() if d["action"] == "BUY"}

                if not phase2_buys:
                    st.info("No underweight assets require accumulation in this cycle.")

                for pair, details in phase2_buys.items():
                    target_val = float(details["target_adjustment_val"])
                    current_price = float(prices.get(pair, 0.0))

                    projected_w = (holdings_val[pair] + target_val) / total_portfolio_val if total_portfolio_val > 0 else 0.0
                    max_allowed = position_caps.get(pair, 0.40)
                    if projected_w > max_allowed:
                        st.warning(f"⚠️ Skipping BUY for {pair}: Projected weight exceeds max limit ({max_allowed*100}%).")
                        continue

                    # Flexible Self-Funding: Utilize available projected NGN cash without arbitrary 90% artificial throttling
                    allocatable_cash = min(target_val, projected_ngn_cash)

                    if allocatable_cash < min_order_threshold:
                        st.warning(f"⚠️ Skipping BUY for {pair}: Insufficient NGN liquidity (Available: ₦{projected_ngn_cash:,.2f}).")
                        continue

                    if current_price > 0:
                        if order_executor and not paper_mode:
                            res = safe_execute_market_order(order_executor, pair, "BUY", allocatable_cash, current_price)
                            st.write(f"Executed BUY on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash -= allocatable_cash
                            else:
                                err_msg = res.get('error', 'Unknown error')
                                st.warning(f"⚠️ BUY {pair} exchange note: {err_msg}")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed BUY on {pair} for ₦{allocatable_cash:,.2f} funded from NGN reserve.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=allocatable_cash, units="SIMULATED", status="PAPER_BUY")
                            projected_ngn_cash -= allocatable_cash

                st.success("✅ Autonomous Two-Phase Rebalancing cycle completed and recorded to SQLite WAL audit logs.")
                log_audit(
                    venue="SYSTEM",
                    asset="PORTFOLIO",
                    price=0.0,
                    amount=total_portfolio_val,
                    units="ALL",
                    status="TWO_PHASE_REBALANCE_SUCCESS",
                )
        except Exception as e:
            st.error(f"🚨 Execution Error: {str(e)}")
            logger.exception("Rebalance execution failed")
